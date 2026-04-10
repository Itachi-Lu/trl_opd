#!/usr/bin/env bash

set -euo pipefail

export WANDB_PROJECT="${WANDB_PROJECT:-opd_1.7b_base_capybara_sft247_three_gate}"
export WANDB_NAME="${WANDB_NAME:-2026_4_9_three_gate}"

# Paper-style OPD baseline with base-gate + disagreement bonus enabled:
# - training batch size B = 128
# - mini-batch size B_mini = 32
# - samples per prompt = 1
# - max response length = 4096
# - non-thinking chat template during training
#
# This launch assumes 8 GPUs:
# - groups_per_batch * group_size = 128 * 1 = 128 total rollout samples
# - per_device_train_batch_size = 4 => global mini-batch = 4 * 8 = 32
# - steps_per_generation = 128 / 32 = 4
# - gradient_accumulation_steps = 1 => 4 optimizer updates per rollout batch

accelerate launch \
  --config_file /apdcephfs_qy4/share_302593112/shaofanliu/projects/lzh/trl/examples/accelerate_configs/deepspeed_zero3.yaml \
  opd.py \
  --model_name_or_path /apdcephfs_qy4_302593112/share_302593112/shaofanliu/projects/lzh/trl/fan/SFT/output_qwen3_1_7base_capybara/checkpoint-247 \
  --tokenizer_name_or_path Qwen/Qwen3-1.7B \
  --teacher_model_name_or_path Qwen/Qwen3-8B \
  --dataset_name qwedsacf/competition_math \
  --dataset_train_split train \
  --dataset_eval_split none \
  --enable_thinking false \
  --output_dir outputs/opd-qwen3_1_7base_capybarasft-from-8b-paper-three-gate \
  --learning_rate 3e-6 \
  --lr_scheduler_type cosine \
  --num_train_epochs 3 \
  --groups_per_batch 128 \
  --group_size 1 \
  --tinker_single_update false \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --max_completion_length 4096 \
  --temperature 1.0 \
  --top_p 1.0 \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps 100 \
  --save_total_limit 20 \
  --eval_strategy no \
  --report_to wandb \
  --bf16 true \
  --gradient_checkpointing \
  --dtype bfloat16 \
  --use_dual_gate true \
  --use_gate_bonus true \
  --gate_teacher_entropy_lambda 3.0 \
  --gate_student_topk 16 \
  --gate_rank_tau 2.0 \
  --gate_weight_min 0.2 \
  --gate_weight_max 2.0 \
  --gate_bonus_min 1.0 \
  --gate_bonus_max 2.0
