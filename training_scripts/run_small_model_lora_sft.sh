#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_FILE="${TRAIN_FILE:-$ROOT_DIR/outputs/synthetic/sft_data/train__gemma-4-31b-it__full_v2_sft.jsonl}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/logs/sft_small}"
TRAIN_LOG_DIR="$LOG_ROOT/train"
EVAL_LOG_DIR="$LOG_ROOT/eval_test"
START_AT="${START_AT:-}"
STARTED=0

mkdir -p "$TRAIN_LOG_DIR" "$EVAL_LOG_DIR"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

safe_name() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9.-]/_/g; s/^_*//; s/_*$//'
}

run_model() {
  local label="$1"
  local model="$2"
  local per_device_batch_size="$3"
  local gradient_accumulation_steps="$4"

  local model_safe
  model_safe="$(safe_name "$model")"
  local checkpoint_dir="$ROOT_DIR/outputs/sft/checkpoints/${model_safe}__gemma-4-31b-v2-lora"
  local train_log="$TRAIN_LOG_DIR/${model_safe}.log"
  local eval_log="$EVAL_LOG_DIR/${model_safe}__test.log"

  if [[ -e "$checkpoint_dir" ]]; then
    echo "[$(timestamp)] Refusing to overwrite existing checkpoint dir: $checkpoint_dir" >&2
    exit 1
  fi

  echo "[$(timestamp)] Starting train: $label"
  echo "[$(timestamp)] Train log: $train_log"
  python -u "$ROOT_DIR/training_scripts/train_lora_sft.py" \
    --model "$model" \
    --train_file "$TRAIN_FILE" \
    --output_dir "$checkpoint_dir" \
    --epochs 2 \
    --per_device_batch_size "$per_device_batch_size" \
    --gradient_accumulation_steps "$gradient_accumulation_steps" \
    --max_seq_length 1024 \
    --learning_rate 2e-5 \
    --logging_steps 25 \
    2>&1 | tee "$train_log"

  echo "[$(timestamp)] Starting test eval: $label"
  echo "[$(timestamp)] Eval log: $eval_log"
  python -u "$ROOT_DIR/src/run_ablation.py" \
    --engine vllm \
    --model "$model" \
    --lora_adapter "$checkpoint_dir" \
    --split test \
    --generation_max_tokens 32 \
    --vllm_gpu_memory_utilization 0.75 \
    --vllm_max_model_len 4096 \
    --overwrite \
    2>&1 | tee "$eval_log"

  echo "[$(timestamp)] Completed: $label"
}

maybe_run_model() {
  local label="$1"
  local model="$2"
  local per_device_batch_size="$3"
  local gradient_accumulation_steps="$4"

  if [[ -n "$START_AT" && "$STARTED" -eq 0 && "$model" != "$START_AT" ]]; then
    echo "[$(timestamp)] Skipping until START_AT=$START_AT: $label"
    return
  fi

  STARTED=1
  run_model "$label" "$model" "$per_device_batch_size" "$gradient_accumulation_steps"
}

echo "[$(timestamp)] Running small-model LoRA SFT pipeline"
echo "[$(timestamp)] Root dir: $ROOT_DIR"
echo "[$(timestamp)] Train file: $TRAIN_FILE"
if [[ -n "$START_AT" ]]; then
  echo "[$(timestamp)] START_AT: $START_AT"
fi
echo "[$(timestamp)] Python: $(command -v python)"
python --version

maybe_run_model "Qwen 3 4B" "Qwen/Qwen3-4B" 32 1
maybe_run_model "Gemma 3 4B IT" "google/gemma-3-4b-it" 16 2
maybe_run_model "Llama 3.2 3B Instruct" "meta-llama/Llama-3.2-3B-Instruct" 32 1

echo "[$(timestamp)] All small-model LoRA runs completed successfully"
echo "[$(timestamp)] Train logs: $TRAIN_LOG_DIR"
echo "[$(timestamp)] Eval logs: $EVAL_LOG_DIR"
