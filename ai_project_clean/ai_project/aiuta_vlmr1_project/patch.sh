# Patch: trocar .venv por conda activate aiuta em todos os sbatch
for f in slurm/*.sbatch; do
  # Remove linhas com .venv/bin/activate
  sed -i '/\.venv\/bin\/activate/d' "$f"
  # Remove linhas com VENV_ACTIVATE
  sed -i '/VENV_ACTIVATE/d' "$f"
  # Adiciona conda activate depois do "set -x" (ou "set -euo pipefail")
  # Se já não tiver conda activate
  if ! grep -q "conda activate" "$f"; then
    sed -i '/^set -x$/a source ~/miniconda3/etc/profile.d/conda.sh\nconda activate aiuta' "$f"
  fi
done
