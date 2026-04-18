#!/bin/bash
#SBATCH -J bbq_eval_smoke
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 00:45:00
#SBATCH -o logs/eval-smoke-%j.out
#SBATCH -e logs/eval-smoke-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

ENGINE="${ENGINE:-vllm}"
MODEL="${MODEL:-mistralai/Mistral-7B-Instruct-v0.2}"
SPLIT="${SPLIT:-dev}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"

if [[ -n "${PACE_MODULES:-}" ]]; then
  module load ${PACE_MODULES}
fi

if [[ -n "${CONDA_ENV_NAME:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV_NAME"
fi

if [[ -n "${VENV_PATH:-}" ]]; then
  source "$VENV_PATH/bin/activate"
fi

mkdir -p logs outputs

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"
echo "Engine: ${ENGINE}"
echo "Model: ${MODEL}"
echo "Split: ${SPLIT}"
echo "Num samples: ${NUM_SAMPLES}"
date

test -f src/run_ablation.py
test -f "data/${SPLIT}.jsonl"

echo
echo "GPU check"
srun nvidia-smi

echo
echo "Python check"
python --version
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0))
PY

echo
echo "Run evaluator tests"
python -m pytest tests

echo
echo "Run evaluator smoke"
srun python src/run_ablation.py \
  --engine "${ENGINE}" \
  --model "${MODEL}" \
  --split "${SPLIT}" \
  --num_samples "${NUM_SAMPLES}"

echo
echo "Recent outputs"
ls -lh outputs | tail

date
echo "Done"
