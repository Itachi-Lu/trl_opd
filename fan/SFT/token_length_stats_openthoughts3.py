#!/usr/bin/env python3
"""
Token length statistics script for OpenThoughts3 with the same config stack as train_sft_openthoughts3.py.
It does NOT train; it only computes token-length distribution on train samples.

Outputs under training_args.output_dir:
- token_count.txt
- token_count.png
- token_count.pang (same PNG bytes, extension kept for compatibility)
"""

import os
import shutil

import matplotlib

matplotlib.use("Agg")

import numpy as np
from datasets import load_dataset
from matplotlib import pyplot as plt
from transformers import AutoTokenizer

from trl import ModelConfig, ScriptArguments, SFTConfig, TrlParser


def get_rank_world():
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def convert_role_values(example):
    role_mapping = {
        "human": "user",
        "gpt": "assistant",
        "system": "system",
    }

    messages = []
    for conv in example.get("conversations", []):
        messages.append(
            {
                "role": role_mapping.get(conv.get("from"), conv.get("from")),
                "content": conv.get("value", ""),
            }
        )

    return {
        "messages": messages,
        "difficulty": example.get("difficulty"),
        "source": example.get("source"),
        "domain": example.get("domain"),
    }


def is_valid_conversational_example(example):
    user_max_chars = 20000
    import random

    a = random.random()   # [0, 1)
    if a < 0.998:
        return False

    messages = example.get("messages", [])
    if not isinstance(messages, list) or len(messages) != 2:
        return False

    first, second = messages

    if first.get("role") != "user":
        return False
    if second.get("role") != "assistant":
        return False

    user_content = first.get("content", "")
    assistant_content = second.get("content", "")

    if not isinstance(user_content, str) or not user_content.strip():
        return False
    if not isinstance(assistant_content, str) or not assistant_content.strip():
        return False

    if len(user_content) > user_max_chars:
        return False

    if "</think>" in assistant_content:
        tail = assistant_content.split("</think>")[-1].strip()
        if not tail:
            return False

    return True


def compute_token_length(tokenizer, messages, max_length):
    out = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = out.get("input_ids", [])
    return len(input_ids)


def percentile_dict(lengths):
    q_list = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    return {f"p{q}": float(np.percentile(lengths, q)) for q in q_list}


def write_stats_file(path, *, total_before_filter, total_after_filter, failed_count, lengths, stats, pct):
    with open(path, "w", encoding="utf-8") as f:
        f.write("Token length statistics (train split)\n")
        f.write("=" * 80 + "\n")
        f.write(f"total_before_filter: {total_before_filter}\n")
        f.write(f"total_after_filter:  {total_after_filter}\n")
        f.write(f"tokenize_failed:     {failed_count}\n")
        f.write(f"counted_samples:     {len(lengths)}\n")
        f.write("\n")

        for k in ["min", "max", "mean", "std", "median"]:
            f.write(f"{k}: {stats[k]:.4f}\n")

        f.write("\nPercentiles\n")
        f.write("-" * 80 + "\n")
        for k in ["p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]:
            f.write(f"{k}: {pct[k]:.4f}\n")


def plot_histogram(path, lengths, max_length):
    max_len = int(max(lengths))

    # Keep bins readable and aligned with practical training lengths.
    if max_length <= 4096:
        bin_size = 128
    elif max_length <= 8192:
        bin_size = 256
    else:
        bin_size = 512

    upper = max(max_len, int(max_length))
    bins = np.arange(0, upper + bin_size, bin_size)
    if len(bins) < 2:
        bins = np.array([0, bin_size])

    plt.figure(figsize=(12, 6))
    plt.hist(lengths, bins=bins, edgecolor="black", alpha=0.85)
    plt.xlabel("Token length range")
    plt.ylabel("Sample count")
    plt.title("Train sample token-length distribution")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def main(script_args, training_args, model_args):
    rank, world_size = get_rank_world()

    if rank != 0:
        # Avoid repeated work/logging in distributed launch.
        return

    os.makedirs(training_args.output_dir, exist_ok=True)
    txt_path = os.path.join(training_args.output_dir, "token_count.txt")
    png_path = os.path.join(training_args.output_dir, "token_count.png")
    pang_path = os.path.join(training_args.output_dir, "token_count.pang")

    print("=" * 100)
    print("Token length stats configuration")
    print("=" * 100)
    print(f"Model: {model_args.model_name_or_path}")
    print(f"Dataset: {script_args.dataset_name}")
    print(f"Output directory: {training_args.output_dir}")
    print(f"World size: {world_size}")
    print(f"Max length: {training_args.max_length}")
    print(f"Dataset num proc: {training_args.dataset_num_proc}")
    print("=" * 100)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    with open("./chatemplate_patch/original_chat_template.jinja", "r", encoding="utf-8") as f:
        tokenizer.chat_template = f.read()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading train split: {script_args.dataset_name} ...")
    dataset = load_dataset(
        script_args.dataset_name,
        name=script_args.dataset_config,
        split=script_args.dataset_train_split,
    )
    total_before_filter = len(dataset)

    remove_cols = [c for c in ["conversations"] if c in dataset.column_names]
    dataset = dataset.map(
        convert_role_values,
        num_proc=training_args.dataset_num_proc,
        remove_columns=remove_cols,
    )
    dataset = dataset.filter(is_valid_conversational_example, num_proc=training_args.dataset_num_proc)
    total_after_filter = len(dataset)

    print(f"Dataset size: {total_before_filter} -> {total_after_filter} (after filter)")
    print("Computing token lengths...")

    lengths = []
    failed_count = 0
    max_length = int(training_args.max_length)

    for i in range(total_after_filter):
        try:
            ex = dataset[i]
            token_len = compute_token_length(tokenizer, ex["messages"], max_length=max_length)
            lengths.append(token_len)
        except Exception:
            failed_count += 1

        if i % 5000 == 0:
            print(f"Processed {i}/{total_after_filter}")

    if not lengths:
        raise RuntimeError("No valid token lengths were computed. Please check dataset and chat template.")

    arr = np.array(lengths, dtype=np.int64)
    stats = {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
    }
    pct = percentile_dict(arr)

    write_stats_file(
        txt_path,
        total_before_filter=total_before_filter,
        total_after_filter=total_after_filter,
        failed_count=failed_count,
        lengths=lengths,
        stats=stats,
        pct=pct,
    )
    plot_histogram(png_path, lengths, max_length=max_length)

    # Keep a second filename requested by user spelling.
    shutil.copyfile(png_path, pang_path)

    print(f"Wrote stats: {txt_path}")
    print(f"Wrote figure: {png_path}")
    print(f"Wrote figure alias: {pang_path}")


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args, _ = parser.parse_args_and_config(return_remaining_strings=True)
    main(script_args, training_args, model_args)
