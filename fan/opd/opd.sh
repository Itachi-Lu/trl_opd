#!/usr/bin/env bash

set -euo pipefail

export WANDB_PROJECT="${WANDB_PROJECT:-cookbook_distillation}"

# If you switch to --tinker_single_update false, also pass:
#   --per_device_train_batch_size <local_batch>
#   --gradient_accumulation_steps <accum_steps>

accelerate launch \
  --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
  fdan/opd/opd.py \
  --model_name_or_path Qwen/Qwen3-1.7B-Base \
  --teacher_model_name_or_path Qwen/Qwen3-4B \
  --dataset_name trl-lib/DeepMath-103K \
  --dataset_train_split train \
  --dataset_eval_split none \
  --output_dir outputs/opd-qwen3-1.7b-base-from-4b \
  --learning_rate 1e-4 \
  --max_steps 100 \
  --groups_per_batch 512 \
  --group_size 4 \
  --tinker_single_update true \
  --max_completion_length 4096 \
  --temperature 1.0 \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps 20 \
  --save_total_limit 2 \
  --eval_strategy no \
  --report_to wandb \
  --bf16 \
  --gradient_checkpointing \
  --use_peft \
  --lora_r 128 \
  --dtype bfloat16
