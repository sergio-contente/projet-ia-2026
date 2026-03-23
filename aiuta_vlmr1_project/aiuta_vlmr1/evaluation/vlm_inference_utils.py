"""Shared VLM inference utilities for offline benchmarks (entropy, timing)."""
from __future__ import annotations

import math
import re
import time

import torch


def compute_first_token_entropy(outputs, vocab_size: int | None = None) -> float:
    """
    Normalized Shannon entropy of the first generated token (CoIN-style).
    Returns float in [0, 1], or -1.0 if unavailable.
    """
    if not hasattr(outputs, "scores") or not outputs.scores:
        return -1.0
    first_logits = outputs.scores[0][0]
    probs = torch.softmax(first_logits.float(), dim=-1)
    log_probs = torch.log2(probs + 1e-10)
    entropy = -(probs * log_probs).sum().item()
    vs = probs.shape[0] if vocab_size is None else vocab_size
    max_entropy = math.log2(vs)
    return entropy / max_entropy if max_entropy > 0 else 0.0


def estimate_reasoning_certainty(reasoning: str) -> str:
    """Coarse certainty bucket from reasoning text."""
    reasoning_lower = reasoning.lower()
    hedging = [
        "appears", "seems", "might", "possibly", "likely",
        "probably", "could be", "looks like", "may be",
        "i'm not sure", "it's hard to tell", "difficult to determine",
        "unclear", "cannot determine", "not certain",
    ]
    confident = [
        "clearly", "definitely", "certainly", "obviously",
        "it is", "i can see", "the image shows",
    ]
    n_hedge = sum(1 for h in hedging if h in reasoning_lower)
    n_confident = sum(1 for c in confident if c in reasoning_lower)
    if n_hedge >= 2:
        return "low"
    if n_hedge >= 1 and n_confident == 0:
        return "low"
    if n_confident >= 2:
        return "high"
    return "medium"


def extract_answer_and_reasoning(raw_output: str) -> tuple[str, str]:
    """Parse <answer> and </think> blocks from VLM-R1 style output."""
    ans_m = re.search(r"<answer>(.*?)</answer>", raw_output, flags=re.DOTALL | re.IGNORECASE)
    raw_answer = ans_m.group(1).strip() if ans_m else raw_output.strip()
    think_m = re.search(r"<think>(.*?)</think>", raw_output, flags=re.DOTALL | re.IGNORECASE)
    reasoning = think_m.group(1).strip() if think_m else ""
    return raw_answer, reasoning


def model_first_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def compute_first_token_max_prob(outputs) -> float | None:
    """Max softmax probability of the first generated token, in (0, 1], or None if unavailable."""
    if not hasattr(outputs, "scores") or not outputs.scores:
        return None
    first_logits = outputs.scores[0][0]
    probs = torch.softmax(first_logits.float(), dim=-1)
    return float(probs.max().item())
