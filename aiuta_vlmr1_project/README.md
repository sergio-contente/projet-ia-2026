# AIUTA-VLM-R1: Reasoning-Enhanced Object Navigation

Integration of VLM-R1 (RL-trained OVD with chain-of-thought reasoning)
and Knowledge Graphs into the AIUTA/CoIN navigation pipeline.

## Quick Start

```bash
pip install -e .
pytest tests/ -v
sbatch slurm/run_coin_vlmr1.sbatch
```

## IDKVQA Benchmark Modes

Primary offline benchmark entrypoint:

```bash
python -m aiuta_vlmr1.evaluation.idkvqa_eval \
  --config configs/idkvqa_eval.yaml \
  --mode raw \
  --limit 20 \
  --output results/idkvqa/smoke_raw.json
```

Supported `--mode` values:

- `raw`: single VQA pass, normalized label only.
- `raw_two_pass`: detection + two VQA passes, no KG fusion.
- `threshold`: `raw` plus uncertainty gate (entropy/max-prob fallback policy).
- `kg`: detection reasoning -> KG attributes -> strict hybrid fusion.
- `kg_threshold`: `kg` plus uncertainty gate.
- `two_pass_kg`: detection + attribute pass + VQA with strict KG fallback.
- `two_pass_kg_relaxed`: same pipeline as `two_pass_kg`, but trusts VLM when no KG slot exists and no hedging is detected.
- `two_pass_kg_entropy`: same as `two_pass_kg_relaxed`, but the final fallback trusts VLM only when `entropy < tau`; otherwise returns `I don't know`.

Notes:

- `tau` comes from `idkvqa_eval.entropy_threshold` (default tuned to `0.09` in `configs/idkvqa_eval.yaml` and `configs/two_pass.yaml`).
- `two_pass_kg` behavior is preserved for backward-compatible comparisons.

## IDKVQA Commands

Run strict two-pass KG:

```bash
python -m aiuta_vlmr1.evaluation.idkvqa_eval \
  --config configs/two_pass.yaml \
  --mode two_pass_kg \
  --output results/idkvqa/two_pass_kg.json
```

Run relaxed two-pass KG:

```bash
python -m aiuta_vlmr1.evaluation.idkvqa_eval \
  --config configs/two_pass.yaml \
  --mode two_pass_kg_relaxed \
  --output results/idkvqa/two_pass_kg_relaxed.json
```

Run entropy-gated relaxed mode:

```bash
python -m aiuta_vlmr1.evaluation.idkvqa_eval \
  --config configs/two_pass.yaml \
  --mode two_pass_kg_entropy \
  --output results/idkvqa/two_pass_kg_entropy.json
```

Run threshold sweep export:

```bash
python -m aiuta_vlmr1.evaluation.idkvqa_eval \
  --config configs/idkvqa_eval.yaml \
  --mode kg_threshold \
  --output results/idkvqa/kg_threshold.json \
  --export-threshold-sweep results/idkvqa/kg_threshold_sweep.json
```

SLURM wrappers:

- `slurm/run_idkvqa.sbatch`: run a single IDKVQA mode (`MODE=...`) with optional second-pass model overrides.
- `slurm/run_idkvqa_ab_compare.sbatch`: compare two outputs and export A/B deltas + threshold sweeps.

## Architecture

```
Observation -> VLMr1Detector (1 call) -> Detection + <think> reasoning
  -> TripleExtractor -> SceneKnowledgeGraph (0 calls)
  -> GraphMatcher -> alignment score (0 calls)
  -> QuestionGenerator -> targeted question (0 calls)
  -> Ask human OR stop OR continue
```

Original AIUTA: 5-8 VLM/LLM calls per detection
Ours: 1 VLM call per detection

## References

- VLM-R1: https://github.com/om-ai-lab/VLM-R1
- CoIN: https://github.com/intelligolabs/CoIN
- Model: https://huggingface.co/omlab/VLM-R1-Qwen2.5VL-3B-OVD-0321
