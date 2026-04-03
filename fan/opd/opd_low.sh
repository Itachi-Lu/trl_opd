#!/usr/bin/env bash

set -euo pipefail

export WANDB_PROJECT="${WANDB_PROJECT:-cookbook_distillation}"

accelerate launch \
  opd.py \
  --model_name_or_path Qwen/Qwen3-1.7B-Base \
  --teacher_model_name_or_path Qwen/Qwen3-4B \
  --dataset_name trl-lib/DeepMath-103K \
  --dataset_train_split train \
  --dataset_eval_split none \
  --enable_thinking true \
  --output_dir outputs/opd-qwen3-1.7b-base-from-4b-test \
  --learning_rate 1e-4 \
  --max_steps 100 \
  --groups_per_batch 64 \
  --group_size 4 \
  --tinker_single_update false \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --max_completion_length 2048 \
  --temperature 1.0 \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps 20 \
  --save_total_limit 2 \
  --eval_strategy no \
  --report_to wandb \
  --bf16 true \
  --gradient_checkpointing \
  --use_peft \
  --lora_r 128 \
  --dtype bfloat16
