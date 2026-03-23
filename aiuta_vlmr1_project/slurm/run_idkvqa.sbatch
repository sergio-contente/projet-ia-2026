#!/bin/bash
#SBATCH --job-name=idkvqa_eval
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=ENSTA-h100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

set -euo pipefail
set -x
whoami && hostname && pwd

cd "$HOME/ai_project/aiuta_vlmr1_project"
source "$HOME/ai_project/.venv/bin/activate"

MODE="${MODE:-raw}"

mkdir -p "$HOME/hf_cache"
export HF_HOME="$HOME/hf_cache"
export HF_HUB_CACHE="$HOME/hf_cache"
export TRANSFORMERS_CACHE="$HOME/hf_cache"

if [ -f "$HOME/.hf_env" ]; then source "$HOME/.hf_env"; fi
if [ -z "${HF_TOKEN:-}" ]; then echo "ERROR: HF_TOKEN not set"; exit 1; fi

pip install datasets --break-system-packages -q 2>/dev/null || true

mkdir -p results/logs results/idkvqa

echo "=== RUN IDKVQA EVALUATION (mode=${MODE}) ==="
python3 -m aiuta_vlmr1.evaluation.idkvqa_eval \
  --config ${CONFIG:-configs/idkvqa_eval.yaml} \
  --mode "$MODE" \
  --output "results/idkvqa/${MODE}_${SLURM_JOB_ID}.json"

echo "=== DONE ==="
