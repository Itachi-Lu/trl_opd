#!/usr/bin/env python3
"""
Token length statistics script for messages-format datasets (e.g., trl-lib/Capybara).
Aligned with train_sft_capybara_no_filter.py:
- No dataset conversion
- No dataset filtering
- Keep the same chat_template patch

Outputs under training_args.output_dir:
- token_count.txt
- token_count.png
- token_count.pang (same PNG bytes, extension kept for compatibility)

python ./token_length_stats_capybara_no_filter.py \
  --model_name_or_path Qwen/Qwen3-1.7B-Base \
  --dataset_name trl-lib/Capybara \
  --dataset_train_split train \
  --max_length 14450 \
  --dataset_num_proc 32 \
  --output_dir ./output_token_stats_capybara

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


def write_stats_file(path, *, total_samples, failed_count, lengths, stats, pct):
    with open(path, "w", encoding="utf-8") as f:
        f.write("Token length statistics (train split)\n")
        f.write("=" * 80 + "\n")
        f.write(f"total_samples:       {total_samples}\n")
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
    total_samples = len(dataset)

    print(f"Dataset size (no filter): {total_samples}")
    print("Computing token lengths...")

    lengths = []
    failed_count = 0
    max_length = int(training_args.max_length)

    for i in range(total_samples):
        try:
            ex = dataset[i]
            messages = ex.get("messages")
            if not isinstance(messages, list):
                raise ValueError("missing or invalid 'messages' field")
            token_len = compute_token_length(tokenizer, messages, max_length=max_length)
            lengths.append(token_len)
        except Exception:
            failed_count += 1

        if i % 5000 == 0:
            print(f"Processed {i}/{total_samples}")

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
        total_samples=total_samples,
        failed_count=failed_count,
        lengths=lengths,
        stats=stats,
        pct=pct,
    )
    plot_histogram(png_path, lengths, max_length=max_length)

    shutil.copyfile(png_path, pang_path)

    print(f"Wrote stats: {txt_path}")
    print(f"Wrote figure: {png_path}")
    print(f"Wrote figure alias: {pang_path}")


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args, _ = parser.parse_args_and_config(return_remaining_strings=True)
    main(script_args, training_args, model_args)
