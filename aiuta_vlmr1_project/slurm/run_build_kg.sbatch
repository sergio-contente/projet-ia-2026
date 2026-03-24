#!/bin/bash
#SBATCH --job-name=build_global_kg
#SBATCH --output=results/logs/build_global_kg-%j.out
#SBATCH --error=results/logs/build_global_kg-%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=ENSTA-h100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

set -euo pipefail
set -x
whoami && hostname && pwd

cd "${KG_PROJECT_ROOT:-$HOME/ai_project/aiuta_vlmr1_project}"
if [ -f "${VENV_ACTIVATE:-$HOME/ai_project/.venv/bin/activate}" ]; then
  # shellcheck source=/dev/null
  source "${VENV_ACTIVATE:-$HOME/ai_project/.venv/bin/activate}"
fi

mkdir -p "$HOME/hf_cache"
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HOME/hf_cache}"

if [ -f "$HOME/.hf_env" ]; then source "$HOME/.hf_env"; fi

pip install datasets --break-system-packages -q 2>/dev/null || true

mkdir -p results/logs results

OUT="${GLOBAL_KG_OUTPUT:-results/global_kg_${SLURM_JOB_ID}.json}"

ARGS=(
  python3 -m aiuta_vlmr1.knowledge_graph.build_global_kg
  --config "${CONFIG:-configs/idkvqa_eval.yaml}"
  --split "${SPLIT:-val}"
  --seed "${SEED:-42}"
  --output "$OUT"
)
if [ -n "${LIMIT_ARG:-}" ]; then
  ARGS+=(--limit "$LIMIT_ARG")
fi

echo "=== BUILD GLOBAL KG → ${OUT} (LIMIT_ARG=${LIMIT_ARG:-}) ==="
"${ARGS[@]}"

echo "=== DONE ==="
