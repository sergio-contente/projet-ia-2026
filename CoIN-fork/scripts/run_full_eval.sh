#!/bin/bash
#SBATCH --job-name=vlmr1_full_eval
#SBATCH --output=logs/vlmr1_full_eval_%j.log
#SBATCH --error=logs/vlmr1_full_eval_%j.log
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate coin-hab

COIN_FORK=/home/ensta/ensta-magalhaes/ai_project/CoIN-fork
AIUTA_PROJECT=/home/ensta/ensta-magalhaes/ai_project/aiuta_vlmr1_project

cd $COIN_FORK
mkdir -p logs results

export PYTHONPATH=$AIUTA_PROJECT:$PYTHONPATH
export TORCH_LIB_DIR=$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')
export LD_LIBRARY_PATH=$TORCH_LIB_DIR:$LD_LIBRARY_PATH
export HF_HOME=/home/ensta/ensta-magalhaes/hf_cache
export HF_HUB_CACHE=$HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME
source ~/.hf_env

export COIN_USE_VLMR1=1
export AIUTA_VLMR1_CONFIG=$AIUTA_PROJECT/configs/vlmr1_coin.yaml
export COIN_VLMR1_DETECT_EVERY=5
export SAM_PORT=12183

# Start MobileSAM
echo "=== Starting MobileSAM ==="
python -m vlfm.vlm.sam --port $SAM_PORT &
SAM_PID=$!
sleep 10
kill -0 $SAM_PID && echo "MobileSAM running (PID=$SAM_PID)"
trap "echo 'Stopping SAM...'; kill $SAM_PID 2>/dev/null || true" EXIT

# --- val_seen ---
echo ""
echo "========================================"
echo "=== SPLIT: val_seen (full) ============"
echo "========================================"
python -m vlfm.run \
  habitat.task.measurements.success.success_distance=0.25 \
  habitat_baselines.eval.split=val_seen \
  habitat.dataset.data_path=CoIN-Bench/val_seen/val_seen.json.gz \
  habitat_baselines.test_episode_count=-1 \
  2>&1 | tee logs/vlmr1_val_seen_full.log

echo "=== val_seen DONE ==="

# --- val_unseen ---
echo ""
echo "========================================"
echo "=== SPLIT: val_unseen (full) =========="
echo "========================================"
python -m vlfm.run \
  habitat.task.measurements.success.success_distance=0.25 \
  habitat_baselines.eval.split=val_unseen \
  habitat.dataset.data_path=CoIN-Bench/val_unseen/val_unseen.json.gz \
  habitat_baselines.test_episode_count=-1 \
  2>&1 | tee logs/vlmr1_val_unseen_full.log

echo "=== val_unseen DONE ==="
echo "=== FULL EVAL COMPLETE ==="
