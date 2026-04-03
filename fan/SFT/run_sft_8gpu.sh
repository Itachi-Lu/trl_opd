#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# SFT Training Script for OpenThoughts3 Dataset
# 8-GPU Distributed Training with LoRA
# Aligned to Thinking Machines SFT setup:
# - Qwen/Qwen3-8B-Base
# - LoRA rank 128
# - LR 1e-3
# - Global batch size 128
# - 3000 steps
# ============================================================================

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-/apdcephfs_qy4/share_302593112/shaofanliu/projects/lzh/trl/fan/SFT/train_sft_openthoughts3.py}"

# GPU Configuration
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29532}"
CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"

# Model Configuration
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B-Base}"
LORA_RANK="${LORA_RANK:-128}"
LORA_ALPHA="${LORA_ALPHA:-256}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"

# Dataset Configuration
DATASET_NAME="${DATASET_NAME:-open-thoughts/OpenThoughts3-1.2M}"
MAX_LENGTH="${MAX_LENGTH:-16384}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-32}"

# Training Configuration
OUTPUT_DIR="${OUTPUT_DIR:-/apdcephfs_qy4/share_302593112/shaofanliu/projects/lzh/trl/fan/SFT/output_qwen3_1_7base}"
MAX_STEPS="${MAX_STEPS:-6000}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"

# Logging and Saving
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-600}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-20}"
REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-qwen3_1.7base_sft}"
SEED="${SEED:-42}"

# Debug
DEBUG_INSPECT_ONLY="${DEBUG_INSPECT_ONLY:-0}"

# Export environment variables
export CUDA_VISIBLE_DEVICES
export CUDA_LAUNCH_BLOCKING
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT
export DEBUG_INSPECT_ONLY

mkdir -p "${OUTPUT_DIR}"

RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${OUTPUT_DIR}/train_${RUN_TIMESTAMP}.log"

echo "========================================"
echo "SFT Training Configuration"
echo "========================================"
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET_NAME}"
echo "Max Length: ${MAX_LENGTH}"
echo "Dataset Num Proc: ${DATASET_NUM_PROC}"
echo "Output: ${OUTPUT_DIR}"
echo "Max Steps: ${MAX_STEPS}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "LoRA Rank: ${LORA_RANK}"
echo "Batch Size (per device): ${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "Gradient Accumulation: ${GRADIENT_ACCUMULATION_STEPS}"
echo "Effective Batch Size: $((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NPROC_PER_NODE))"
echo "GPUs: ${NPROC_PER_NODE}"
echo "Attention implementation: ${ATTN_IMPL}"
echo "DEBUG_INSPECT_ONLY: ${DEBUG_INSPECT_ONLY}"
echo "Log File: ${LOG_FILE}"
echo "========================================"

{
"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --master_port "${MASTER_PORT}" \
  "${TRAIN_SCRIPT}" \
  --model_name_or_path "${MODEL_NAME}" \
  --dataset_name "${DATASET_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_steps "${MAX_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --logging_steps "${LOGGING_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --report_to "${REPORT_TO}" \
  --seed "${SEED}" \
  --max_length "${MAX_LENGTH}" \
  --dataset_num_proc "${DATASET_NUM_PROC}" \
  --use_peft \
  --assistant_only_loss true \
  --lora_r "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --dtype bfloat16 \
  --bf16 true \
  --gradient_checkpointing true \
  --attn_implementation "${ATTN_IMPL}"
} 2>&1 | tee -a "${LOG_FILE}"

echo "========================================"
echo "Training completed!"
echo "Output saved to: ${OUTPUT_DIR}"
echo "Log file: ${LOG_FILE}"
echo "========================================"
