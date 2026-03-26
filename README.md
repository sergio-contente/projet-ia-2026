<h1 align="center">
    AIUTA-VLM-R1: Uncertainty-Aware VQA with Knowledge Graphs
</h1>

<p align="center">
    <b>AI Project -- MVA + ENSTA Paris, 2025-2026</b>
</p>

<p align="center">
<img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/VLM--R1-Qwen2.5VL--3B-green?style=for-the-badge"/>
<!-- ALL-CONTRIBUTORS-BADGE:START -->
<img src="https://img.shields.io/badge/contributors-2-orange.svg?style=for-the-badge"/>
<!-- ALL-CONTRIBUTORS-BADGE:END -->
</p>

<p align="center">
<img src="https://forthebadge.com/images/badges/made-with-python.svg"/>
<img src="https://forthebadge.com/images/badges/built-with-science.svg"/>
</p>

This project integrates [VLM-R1](https://github.com/om-ai-lab/VLM-R1) (an RL-trained Vision Language Model with chain-of-thought reasoning) and **Knowledge Graphs** into the [AIUTA/CoIN](https://github.com/intelligolabs/CoIN) uncertainty-aware VQA pipeline. We extract structured knowledge from VLM reasoning blocks at **zero extra cost** to build scene Knowledge Graphs that inform when the model should abstain ("I don't know") vs. commit to an answer.

---

## Overview

The project addresses two main questions:

1. **Uncertainty-Aware VQA** -- Given an image and a Yes/No question, can we reliably decide when to answer and when to abstain ("I don't know"), minimizing overconfident wrong answers?
2. **Embodied Object Navigation** -- Can a VLM-R1-powered agent find target objects in 3D scenes while minimizing human-agent interaction?

## Architecture

```
Image + Question
  |
  +-- Detection Pass (1 call) --> <think> reasoning --> TripleExtractor --> KG attributes
  |
  +-- Attribute Pass (1 call) --> structured JSON --> merged into KG
  |
  +-- VQA Pass (1 call) --> Yes/No/IDK + token entropy
                                    |
                    +---------------+----------------+
                    |               |                |
              KG Hybrid        Entropy Gate      Hedging Detection
              Fusion           (tau = 0.09)      (textual cues)
                    |               |                |
                    +---------------+----------------+
                                    |
                              Final Answer
                          (Yes / No / I don't know)
```

**Cost**: 1-3 VLM calls per sample (vs. 5-8 in original AIUTA)

## Project Structure

```
projet-ia-2026/
+-- README.md
+-- ai_project/
|   +-- aiuta_vlmr1_project/             # Core project
|   |   +-- aiuta_vlmr1/                 # Main package
|   |   |   +-- config.py                # YAML-driven configuration (Strategy pattern)
|   |   |   +-- detector/                # VLM-R1 object detector
|   |   |   |   +-- vlmr1_detector.py    # VLM-R1 detection adapter
|   |   |   |   +-- output_parser.py     # Parse detections + bboxes from VLM
|   |   |   |   +-- prompt_templates.py  # Prompts for open-vocabulary detection
|   |   |   +-- knowledge_graph/         # Scene KG construction & querying
|   |   |   |   +-- schema.py            # Certainty, Attribute, ObjectNode, SpatialRelation
|   |   |   |   +-- scene_graph.py       # Per-episode KG with object dedup & merging
|   |   |   |   +-- triple_extractor.py  # Extract (subj, pred, obj) from <think> blocks
|   |   |   |   +-- attribute_parser.py  # Parse structured JSON attributes
|   |   |   |   +-- graph_matcher.py     # Alignment scoring (detection vs. target)
|   |   |   |   +-- build_global_kg.py   # Aggregate scene KGs into global index
|   |   |   +-- self_questioner/         # Two-pass attribute extraction
|   |   |   |   +-- two_pass_questioner.py  # Detection + attribute JSON (2 calls)
|   |   |   |   +-- vlmr1_questioner.py     # Single-pass: triples only (1 call)
|   |   |   +-- interaction_trigger/     # Decision logic (ask / continue / stop)
|   |   |   |   +-- kg_trigger.py        # KG-based trigger (0 LLM calls)
|   |   |   +-- evaluation/              # Benchmarking & metrics
|   |   |   |   +-- idkvqa_eval.py       # Primary benchmark entry point
|   |   |   |   +-- idkvqa_kg.py         # KG hybrid fusion (conservative/relaxed/entropy)
|   |   |   |   +-- uncertainty_abstention.py  # Entropy-based threshold gates
|   |   |   |   +-- threshold_sweep.py   # Offline entropy tau optimization
|   |   |   |   +-- paper_artifacts.py   # CSV/JSON exports for paper tables
|   |   |   +-- pipeline/               # Main orchestration
|   |   |   |   +-- aiuta_pipeline.py    # Strategy-pattern component selection
|   |   |   |   +-- coin_bench_runner.py # Bulk benchmark runner
|   |   |   +-- utils/
|   |   |       +-- model_loader.py      # Singleton model cache (Qwen-VL, etc.)
|   |   +-- configs/
|   |   |   +-- idkvqa_eval.yaml         # Default eval config (tau=0.09)
|   |   |   +-- two_pass.yaml            # Two-pass pipeline config
|   |   |   +-- vlmr1_coin.yaml          # CoIN embodied task config
|   |   +-- tests/                       # 19 test modules (no GPU needed)
|   |   +-- slurm/                       # SLURM job submission scripts
|   |   +-- requirements.txt
|   |   +-- pyproject.toml
|   |   +-- setup.py
|   +-- CoIN-fork/                       # Fork of CoIN (VLFM-based embodied agent)
|   +-- CoIN-Bench/                      # Benchmark dataset (HuggingFace)
|   +-- GroundingDINO/                   # Alternative detector
+-- .gitignore
```

## Ablation Modes

| Mode | VLM Calls | Description |
|------|-----------|-------------|
| `raw` | 1 | VLM answer only (baseline) |
| `threshold` | 1 | raw + entropy-based abstention |
| `kg` | 2 | Detection reasoning -> KG -> hybrid fusion (conservative) |
| `kg_threshold` | 2 | KG + entropy gate |
| `two_pass_kg` | 3 | KG + structured attribute pass (conservative fusion) |
| `two_pass_kg_relaxed` | 3 | Same pipeline, trust VLM when confident |
| `two_pass_kg_entropy` | 3 | Same pipeline, trust VLM only when entropy < tau |
| `global_kg` | 1 | Pre-built global KG lookup (no live detection) |
| `global_kg_entropy` | 1 | Global KG + entropy gate |

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Primary VLM | `omlab/VLM-R1-Qwen2.5VL-3B-OVD-0321` |
| Processor | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Precision | bfloat16 |
| Entropy threshold (tau) | 0.09 |
| Abstention rule | entropy_above_tau_to_idk |
| VQA max tokens | 256 |
| Detection max tokens | 512 |
| Seed | 42 |

## Primary Benchmark: IDKVQA

Evaluated on [ftaioli/IDKVQA](https://huggingface.co/datasets/ftaioli/IDKVQA) (502 samples, val split).

Key metrics:
- **Effective Reliability (phi_c)**: penalizes confident wrong answers more than abstentions
- **Accuracy**: fraction of correct predictions
- **Overclaim rate**: saying Yes/No when ground truth is IDK
- **Underclaim rate**: saying IDK when ground truth is Yes/No
- **Coverage**: fraction of non-IDK predictions

## Datasets

| Dataset | Role | Source |
|---------|------|--------|
| IDKVQA | Primary offline VQA benchmark (502 samples) | [ftaioli/IDKVQA](https://huggingface.co/datasets/ftaioli/IDKVQA) |
| CoIN-Bench | Embodied navigation benchmark (val_seen, val_unseen, val_seen_synonyms) | [ftaioli/CoIN-Bench](https://huggingface.co/datasets/ftaioli/CoIN-Bench) |

## Usage

### Installation

```bash
git clone https://github.com/sergio-contente/projet-ia-2026.git
cd projet-ia-2026/ai_project/aiuta_vlmr1_project
pip install -e .
```

### Requirements

```bash
pip install torch>=2.1 transformers>=4.40 qwen-vl-utils accelerate networkx>=3.0 pyyaml>=6.0 Pillow>=10.0 numpy>=1.24 matplotlib>=3.7 pytest>=7.0
```

### Running Tests (no GPU needed)

```bash
cd ai_project/aiuta_vlmr1_project
pytest tests/ -v
```

### Running IDKVQA Evaluation (requires GPU)

```bash
# Via SLURM
sbatch --export=ALL,MODE=two_pass_kg_entropy slurm/run_idkvqa.sbatch

# Other modes
sbatch --export=ALL,MODE=raw slurm/run_idkvqa.sbatch
sbatch --export=ALL,MODE=kg slurm/run_idkvqa.sbatch
sbatch --export=ALL,MODE=two_pass_kg_relaxed slurm/run_idkvqa.sbatch

# Direct CLI
python -m aiuta_vlmr1.evaluation.idkvqa_eval \
  --config configs/idkvqa_eval.yaml \
  --mode two_pass_kg_entropy \
  --output results/idkvqa/run.json

# Smoke test (5 samples)
sbatch --export=ALL,MODE=two_pass_kg_entropy,LIMIT_ARG=5 slurm/run_idkvqa.sbatch
```

### A/B Comparison (no GPU)

```bash
python3 -m aiuta_vlmr1.evaluation.idkvqa_ab_compare \
  --input-a results/idkvqa/run_a.json \
  --input-b results/idkvqa/run_b.json \
  --label-a baseline --label-b improved \
  --output results/idkvqa/ab_compare.json
```

## Key Idea

VLM-R1 produces `<think>` reasoning blocks before answering. We parse these blocks with regex-based triple extraction -- **at zero extra GPU cost** -- to populate a scene Knowledge Graph. The KG then drives:

1. **Conservative fusion**: only confirm when KG attributes match the target
2. **Entropy gating**: abstain when token entropy exceeds tau=0.09
3. **Hedging detection**: flag textual uncertainty cues ("might", "possibly")

This reduces overconfident wrong answers (overclaiming) at the cost of slightly higher abstention.

## References

- **VLM-R1**: [om-ai-lab/VLM-R1](https://github.com/om-ai-lab/VLM-R1) -- RL-trained open-vocabulary detection
- **CoIN/AIUTA**: [intelligolabs/CoIN](https://github.com/intelligolabs/CoIN) -- Cooperative Interaction Agent
- **IDKVQA**: [ftaioli/IDKVQA](https://huggingface.co/datasets/ftaioli/IDKVQA) -- Yes/No/IDK visual QA
- **Model**: [omlab/VLM-R1-Qwen2.5VL-3B-OVD-0321](https://huggingface.co/omlab/VLM-R1-Qwen2.5VL-3B-OVD-0321)
- **VLM-R1 Blog**: [om-ai-lab.github.io](https://om-ai-lab.github.io/2025_03_20.html)
- **CoIN Project Page**: [intelligolabs.github.io/CoIN](https://intelligolabs.github.io/CoIN/)

## Contributors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/sergio-contente">
        <img src="https://avatars.githubusercontent.com/u/78960602?v=4" width="100px;" alt=""/><br/>
        <sub><b>Sergio Magalhaes Contente</b></sub>
      </a>
    </td>
  </tr>
</table>
