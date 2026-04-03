#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Token Length Statistics Script for OpenThoughts3 Dataset
# This script only computes token-length distribution and writes:
# - token_count.txt
# - token_count.png
# - token_count.pang
# ============================================================================

PYTHON_BIN="${PYTHON_BIN:-python}"
STATS_SCRIPT="${STATS_SCRIPT:-/apdcephfs_qy4/share_302593112/shaofanliu/projects/lzh/trl/fan/SFT/token_length_stats_openthoughts3.py}"

# Runtime Configuration
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"

# Model Configuration
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B-Base}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"

# Dataset Configuration
DATASET_NAME="${DATASET_NAME:-open-thoughts/OpenThoughts3-1.2M}"
MAX_LENGTH="${MAX_LENGTH:-32768}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-64}"

# Output / Logging
OUTPUT_DIR="${OUTPUT_DIR:-/apdcephfs_qy4/share_302593112/shaofanliu/projects/lzh/trl/fan/SFT/output}"
REPORT_TO="${REPORT_TO:-none}"
SEED="${SEED:-42}"

export CUDA_VISIBLE_DEVICES
export CUDA_LAUNCH_BLOCKING
export TOKENIZERS_PARALLELISM=false

mkdir -p "${OUTPUT_DIR}"

RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${OUTPUT_DIR}/token_stats_${RUN_TIMESTAMP}.log"

echo "========================================"
echo "Token Length Statistics Configuration"
echo "========================================"
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET_NAME}"
echo "Max Length: ${MAX_LENGTH}"
echo "Dataset Num Proc: ${DATASET_NUM_PROC}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs (visible): ${CUDA_VISIBLE_DEVICES}"
echo "Attention implementation: ${ATTN_IMPL}"
echo "Stats Script: ${STATS_SCRIPT}"
echo "Log File: ${LOG_FILE}"
echo "========================================"

{
"${PYTHON_BIN}" "${STATS_SCRIPT}" \
  --model_name_or_path "${MODEL_NAME}" \
  --dataset_name "${DATASET_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --report_to "${REPORT_TO}" \
  --seed "${SEED}" \
  --max_length "${MAX_LENGTH}" \
  --dataset_num_proc "${DATASET_NUM_PROC}" \
  --attn_implementation "${ATTN_IMPL}"
} 2>&1 | tee -a "${LOG_FILE}"

echo "========================================"
echo "Token statistics completed!"
echo "Output saved to: ${OUTPUT_DIR}"
echo "Log file: ${LOG_FILE}"
echo "========================================"
