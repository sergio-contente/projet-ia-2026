# AIUTA-VLM-R1: Uncertainty-Aware Visual Question Answering with Reasoning-Enhanced VLMs

Integration of [VLM-R1](https://github.com/om-ai-lab/VLM-R1) (RL-trained VLM with chain-of-thought reasoning) and Knowledge Graphs into the [AIUTA/CoIN](https://github.com/intelligolabs/CoIN) uncertainty-aware VQA pipeline.

## Key Idea

VLM-R1 produces `<think>` reasoning blocks before answering. We extract structured knowledge (triples, attributes) from this reasoning — **at zero extra cost** — to build a scene Knowledge Graph that informs when the model should abstain ("I don't know") vs commit to an answer.

## Architecture

```
Image + Question
  │
  ├─ Detection Pass (1 call) ──→ <think> reasoning ──→ TripleExtractor ──→ KG attributes
  │
  ├─ Attribute Pass (1 call) ──→ structured JSON ──→ merged into KG
  │
  └─ VQA Pass (1 call) ──→ Yes/No/IDK + token entropy
                                    │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
              KG Hybrid        Entropy Gate      Hedging Detection
              Fusion           (τ = 0.09)        (textual cues)
                    │                │                │
                    └────────────────┴────────────────┘
                                    │
                              Final Answer
                          (Yes / No / I don't know)
```

**Cost**: 1–3 VLM calls per sample (vs 5–8 in original AIUTA)

## Ablation Modes

| Mode | Calls | Description |
|------|-------|-------------|
| `raw` | 1 | VLM answer only (baseline) |
| `threshold` | 1 | raw + entropy-based abstention |
| `kg` | 2 | Detection reasoning → KG → hybrid fusion (conservative) |
| `two_pass_kg` | 3 | KG + structured attribute pass (conservative fusion) |
| `two_pass_kg_relaxed` | 3 | Same pipeline, trust VLM when confident |
| `two_pass_kg_entropy` | 3 | Same pipeline, trust VLM only when entropy < τ |

## Quick Start

```bash
# Install
pip install -e .

# Run tests (no GPU needed)
pytest tests/ -v

# Run IDKVQA evaluation (requires GPU)
cd ~/ai_project/aiuta_vlmr1_project
sbatch --export=ALL,MODE=two_pass_kg_entropy slurm/run_idkvqa.sbatch

# Other modes
sbatch --export=ALL,MODE=raw slurm/run_idkvqa.sbatch
sbatch --export=ALL,MODE=kg slurm/run_idkvqa.sbatch
sbatch --export=ALL,MODE=two_pass_kg_relaxed slurm/run_idkvqa.sbatch

# With dedicated second-pass model
sbatch --export=ALL,MODE=two_pass_kg,SECOND_PASS_MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct,SECOND_PASS_PROCESSOR_ID=Qwen/Qwen2.5-VL-3B-Instruct slurm/run_idkvqa.sbatch

# Smoke test (5 samples)
sbatch --export=ALL,MODE=two_pass_kg_entropy,LIMIT_ARG=5 slurm/run_idkvqa.sbatch

# A/B comparison (no GPU)
python3 -m aiuta_vlmr1.evaluation.idkvqa_ab_compare \
  --input-a results/idkvqa/run_a.json \
  --input-b results/idkvqa/run_b.json \
  --label-a baseline --label-b improved \
  --output results/idkvqa/ab_compare.json
```

## Primary Benchmark: IDKVQA

Evaluated on [ftaioli/IDKVQA](https://huggingface.co/datasets/ftaioli/IDKVQA) (502 samples, val split).

Key metrics:
- **Effective Reliability φ(c)**: penalizes confident wrong answers more than abstentions
- **Accuracy**: fraction of correct predictions
- **Overclaim rate**: saying Yes/No when ground truth is IDK
- **Underclaim rate**: saying IDK when ground truth is Yes/No
- **Coverage**: fraction of non-IDK predictions

## Project Structure

```
aiuta_vlmr1/
├── config.py                    # YAML-driven configuration (Strategy pattern)
├── detector/                    # VLM-R1 object detector
├── knowledge_graph/             # Scene KG, triple extraction, attribute parsing
├── self_questioner/             # Two-pass attribute extraction
├── evaluation/
│   ├── idkvqa_eval.py           # Primary benchmark entry point
│   ├── idkvqa_kg.py             # KG hybrid fusion (conservative/relaxed/entropy)
│   ├── idkvqa_ab_compare.py     # A/B comparison tool
│   ├── threshold_sweep.py       # Offline entropy τ sweep
│   ├── mode_transition_analysis.py
│   └── paper_artifacts.py       # CSV/JSON exports for paper tables
├── utils/
│   └── model_loader.py          # Config-keyed model singleton
configs/
├── idkvqa_eval.yaml             # Default eval config (τ=0.09)
├── two_pass.yaml                # Two-pass pipeline config
slurm/
├── run_idkvqa.sbatch            # Main evaluation launcher
scripts/
├── download_models.sh           # Pre-download models to HF cache
```

## References

- **VLM-R1**: [om-ai-lab/VLM-R1](https://github.com/om-ai-lab/VLM-R1) — RL-trained open-vocabulary detection
- **CoIN/AIUTA**: [intelligolabs/CoIN](https://github.com/intelligolabs/CoIN) — Cooperative Interaction Agent
- **IDKVQA**: [ftaioli/IDKVQA](https://huggingface.co/datasets/ftaioli/IDKVQA) — Yes/No/IDK visual QA
- **Model**: [omlab/VLM-R1-Qwen2.5VL-3B-OVD-0321](https://huggingface.co/omlab/VLM-R1-Qwen2.5VL-3B-OVD-0321)
- **VLM-R1 Blog**: [om-ai-lab.github.io](https://om-ai-lab.github.io/2025_03_20.html)
- **CoIN Project Page**: [intelligolabs.github.io/CoIN](https://intelligolabs.github.io/CoIN/)
