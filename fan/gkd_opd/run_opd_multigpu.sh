#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-/apdcephfs_qy4/share_302593112/shaofanliu/projects/lzh/trl/fan/gkd_opd/train_opd_deepscaler_aime2024.py}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29531}"
CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"

STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen3-1.7B-Base}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-4B}"
TRAIN_DATASET="${TRAIN_DATASET:-agentica-org/DeepScaleR-Preview-Dataset}"
EVAL_DATASET="${EVAL_DATASET:-HuggingFaceH4/aime_2024}"
OUTPUT_DIR="${OUTPUT_DIR:-/apdcephfs_qy4/share_302593112/shaofanliu/projects/lzh/trl/fan/gkd_opd/gkd-opd-qwen3-multigpu}"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-10240}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-812}"
TEMPERATURE="${TEMPERATURE:-1.0}"

EVAL_STRATEGY="${EVAL_STRATEGY:-steps}"
EVAL_STEPS="${EVAL_STEPS:-5}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
SAVE_STEPS="${SAVE_STEPS:-50}"
REPORT_TO="${REPORT_TO:-wandb}"
ROLLOUT_DEBUG="${ROLLOUT_DEBUG:-1}"
ROLLOUT_DEBUG_STEPS="${ROLLOUT_DEBUG_STEPS:-50}"
ROLLOUT_DEBUG_NUM_SAMPLES="${ROLLOUT_DEBUG_NUM_SAMPLES:-2}"

ROLLOUT_DEBUG_FLAG=()
if [[ "${ROLLOUT_DEBUG}" == "1" ]]; then
  ROLLOUT_DEBUG_FLAG=(--rollout_debug)
fi

export CUDA_VISIBLE_DEVICES
export CUDA_LAUNCH_BLOCKING
export TOKENIZERS_PARALLELISM=false

mkdir -p "${OUTPUT_DIR}"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${OUTPUT_DIR}/train_${RUN_TIMESTAMP}.log"
echo "[run_opd_multigpu] logging to ${LOG_FILE}"

{
"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --master_port "${MASTER_PORT}" \
  "${TRAIN_SCRIPT}" \
  --student_model "${STUDENT_MODEL}" \
  --teacher_model "${TEACHER_MODEL}" \
  --train_dataset "${TRAIN_DATASET}" \
  --eval_dataset "${EVAL_DATASET}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --learning_rate "${LEARNING_RATE}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --max_seq_length "${MAX_SEQ_LENGTH}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --eval_strategy "${EVAL_STRATEGY}" \
  --eval_steps "${EVAL_STEPS}" \
  --logging_steps "${LOGGING_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --report_to "${REPORT_TO}" \
  --rollout_debug_steps "${ROLLOUT_DEBUG_STEPS}" \
  --rollout_debug_num_samples "${ROLLOUT_DEBUG_NUM_SAMPLES}" \
  "${ROLLOUT_DEBUG_FLAG[@]}"
} 2>&1 | tee -a "${LOG_FILE}"
