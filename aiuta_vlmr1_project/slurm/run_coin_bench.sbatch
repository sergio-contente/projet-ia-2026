#!/bin/bash
#SBATCH --job-name=coin_bench
#SBATCH --partition=ENSTA-l40s
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=results/logs/coin_bench-%j.out
#SBATCH --error=results/logs/coin_bench-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
set -x

cd "${COIN_PROJECT_ROOT:-$HOME/ai_project/aiuta_vlmr1_project}"

if [ -f "${VENV_ACTIVATE:-$HOME/ai_project/.venv/bin/activate}" ]; then
  # shellcheck source=/dev/null
  source "${VENV_ACTIVATE:-$HOME/ai_project/.venv/bin/activate}"
fi

MODE="${MODE:-entropy_nav}"
SPLIT="${SPLIT:-val_seen}"
TAU="${TAU:-0.15}"
GLOBAL_KG_PATH="${GLOBAL_KG_PATH:-}"
LIMIT_ARG="${LIMIT_ARG:-}"
CONFIG="${CONFIG:-configs/idkvqa_eval.yaml}"

mkdir -p "$HOME/hf_cache"
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HOME/hf_cache}"

if [ -f "$HOME/.hf_env" ]; then source "$HOME/.hf_env"; fi

mkdir -p results/logs results/coin_bench

OUT="results/coin_bench/${MODE}_${SLURM_JOB_ID}.json"

EXTRA_ARGS=()
[ -n "$GLOBAL_KG_PATH" ] && EXTRA_ARGS+=(--global-kg-path "$GLOBAL_KG_PATH")
[ -n "$LIMIT_ARG" ] && EXTRA_ARGS+=(--limit "$LIMIT_ARG")

python -m aiuta_vlmr1.pipeline.coin_bench_runner \
  --config "$CONFIG" \
  --coin-bench-path "${COIN_BENCH_PATH:-$HOME/data/coin_bench}" \
  --mode "$MODE" \
  --split "$SPLIT" \
  --tau "$TAU" \
  --output "$OUT" \
  "${EXTRA_ARGS[@]}"
