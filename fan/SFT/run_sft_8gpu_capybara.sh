#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# SFT Training Script for Capybara Dataset
# 8-GPU Distributed Full-Parameter Training (No LoRA)
# - Model: Qwen/Qwen3-1.7B-Base
# - Dataset: trl-lib/Capybara
# - Training length: 1 epoch
# - Keep chat-template patch + assistant_only_loss workflow in train script
# ============================================================================

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-./train_sft_capybara_no_filter.py}"

# GPU Configuration
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29532}"
CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"

# Model Configuration
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B-Base}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"

# Dataset Configuration
DATASET_NAME="${DATASET_NAME:-trl-lib/Capybara}"
MAX_LENGTH="${MAX_LENGTH:-16384}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-32}"

# Training Configuration (full-parameter SFT for instruction following)
OUTPUT_DIR="${OUTPUT_DIR:-./output_qwen3_1_7base_capybara}"
MAX_STEPS="${MAX_STEPS:--1}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"

# Logging and Saving
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-50}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-20}"
REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-qwen3_1.7base_capybara}"
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
echo "Train Script: ${TRAIN_SCRIPT}"
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET_NAME}"
echo "Max Length: ${MAX_LENGTH}"
echo "Dataset Num Proc: ${DATASET_NUM_PROC}"
echo "Output: ${OUTPUT_DIR}"
echo "Max Steps: ${MAX_STEPS}"
echo "Num Train Epochs: ${NUM_TRAIN_EPOCHS}"
echo "Learning Rate: ${LEARNING_RATE}"
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
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
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
  --assistant_only_loss true \
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
