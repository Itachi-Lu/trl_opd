#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# python merge_lora.py \
#   --base_model Qwen/Qwen3-8B-Base \
#   --adapter_path /apdcephfs_qy4/share_302593112/shaofanliu/projects/lzh/trl/fan/SFT/output/checkpoint-6000 \
#   --save_path /apdcephfs_qy4/share_302593112/shaofanliu/projects/lzh/trl/fan/SFT/output/merged-checkpoint-6000 \
#   --dtype bfloat16

from pathlib import Path
import argparse
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-8B-Base")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    adapter_path = Path(args.adapter_path).expanduser().resolve()
    save_path = Path(args.save_path).expanduser().resolve()
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading tokenizer from base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
    )

    print(f"[2/5] Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
        device_map="cpu",
    )

    print(f"[3/5] Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))

    print("[4/5] Merging adapter into base model...")
    merged_model = model.merge_and_unload()

    print(f"[5/5] Saving merged model to: {save_path}")
    merged_model.save_pretrained(str(save_path), safe_serialization=True)
    tokenizer.save_pretrained(str(save_path))

    print("\nDone.")
    print(f"Merged model saved at: {save_path}")


if __name__ == "__main__":
    main()