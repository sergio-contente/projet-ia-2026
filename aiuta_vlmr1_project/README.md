# AIUTA-VLM-R1: Reasoning-Enhanced Object Navigation

Integration of VLM-R1 (RL-trained OVD with chain-of-thought reasoning)
and Knowledge Graphs into the AIUTA/CoIN navigation pipeline.

## Quick Start

```bash
pip install -e .
pytest tests/ -v
sbatch slurm/run_coin_vlmr1.sbatch
```

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
