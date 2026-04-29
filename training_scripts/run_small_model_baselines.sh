#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/logs/baselines_small}"
LOG_DIR="$LOG_ROOT/test"
START_AT="${START_AT:-}"
STARTED=0

mkdir -p "$LOG_DIR"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

safe_name() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9.-]/_/g; s/^_*//; s/_*$//'
}

run_model() {
  local label="$1"
  local model="$2"
  local model_safe
  model_safe="$(safe_name "$model")"
  local log_path="$LOG_DIR/${model_safe}.log"

  echo "[$(timestamp)] Starting baseline test: $label"
  echo "[$(timestamp)] Log: $log_path"
  python -u "$ROOT_DIR/src/run_ablation.py" \
    --engine vllm \
    --model "$model" \
    --split test \
    --generation_max_tokens 32 \
    --vllm_gpu_memory_utilization 0.75 \
    --vllm_max_model_len 4096 \
    --overwrite \
    2>&1 | tee "$log_path"
  echo "[$(timestamp)] Completed baseline test: $label"
}

maybe_run_model() {
  local label="$1"
  local model="$2"

  if [[ -n "$START_AT" && "$STARTED" -eq 0 && "$model" != "$START_AT" ]]; then
    echo "[$(timestamp)] Skipping until START_AT=$START_AT: $label"
    return
  fi

  STARTED=1
  run_model "$label" "$model"
}

echo "[$(timestamp)] Running small-model baseline tests"
echo "[$(timestamp)] Root dir: $ROOT_DIR"
if [[ -n "$START_AT" ]]; then
  echo "[$(timestamp)] START_AT: $START_AT"
fi
echo "[$(timestamp)] Python: $(command -v python)"
python --version

maybe_run_model "Qwen 3 4B" "Qwen/Qwen3-4B"
maybe_run_model "Gemma 3 4B IT" "google/gemma-3-4b-it"
maybe_run_model "Llama 3.2 3B Instruct" "meta-llama/Llama-3.2-3B-Instruct"

echo "[$(timestamp)] All small-model baseline tests completed successfully"
echo "[$(timestamp)] Logs: $LOG_DIR"
