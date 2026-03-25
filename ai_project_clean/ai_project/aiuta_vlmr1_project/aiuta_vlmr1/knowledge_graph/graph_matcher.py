"""
graph_matcher.py — Alignment scoring between detected objects and target facts.
Replaces AIUTA P_score LLM prompt with deterministic graph matching.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any

from collections.abc import Callable

from .schema import ObjectNode, TargetFacts


SYNONYMS: dict[str, str] = {
    "wooden": "wood",
    "hardwood": "wood",
    "metallic": "metal",
    "grey": "gray",
    "big": "large",
    "tiny": "small",
    "dark brown": "brown",
    "light brown": "brown",
    "in the bedroom": "bedroom",
    "in the kitchen": "kitchen",
    "in the bathroom": "bathroom",
    "in the living room": "living_room",
    "bedroom": "bedroom",
    "kitchen": "kitchen",
}


def _norm(s: str) -> str:
    return " ".join(s.strip().lower().split())


def _canonical(val: str) -> str:
    x = _norm(val)
    for key in sorted(SYNONYMS.keys(), key=len, reverse=True):
        if key in x:
            x = x.replace(key, SYNONYMS[key])
    return " ".join(x.split())


def _semantic_match(obj_val: str, target_val: str) -> bool:
    o = _canonical(obj_val)
    t = _canonical(target_val)
    if o == t:
        return True
    if t in o or o in t:
        return True
    return False


class GraphMatcher:
    @staticmethod
    def compute_alignment(obj: ObjectNode, target: TargetFacts) -> float:
        if target.num_facts == 0:
            return -1.0
        total = 0
        matched = 0
        contradicted = 0
        for attr_name, target_val in target.known_attributes.items():
            total += 1
            obj_val = obj.get_attribute_value(attr_name)
            if obj_val is not None:
                if _semantic_match(obj_val, target_val):
                    matched += 1
                else:
                    contradicted += 1
        for attr_name, neg_val in target.negative_attributes.items():
            total += 1
            obj_val = obj.get_attribute_value(attr_name)
            if obj_val is not None:
                if _semantic_match(obj_val, neg_val):
                    contradicted += 1
                else:
                    matched += 1
        if contradicted > 0:
            return 0.0
        if total == 0:
            return -1.0
        return matched / total

    @staticmethod
    def compute_alignment_with_vlm_fallback(
        obj: ObjectNode,
        target: TargetFacts,
        tau_stop: float = 0.8,
        vlm_judge_fn: Callable[[str, str], bool] | None = None,
    ) -> float:
        score = GraphMatcher.compute_alignment(obj, target)
        if score >= tau_stop or score == 0.0:
            return score
        if vlm_judge_fn is None:
            return score
        obj_desc = obj.to_natural_language()
        target_desc = target.to_natural_language()
        try:
            is_match = vlm_judge_fn(obj_desc, target_desc)
            if is_match:
                return tau_stop
        except Exception:
            pass
        return score

    @staticmethod
    def explain_alignment(obj: ObjectNode, target: TargetFacts) -> dict[str, Any]:
        """Structured breakdown for debugging / papers (not a second scoring API)."""
        matched: list[dict[str, str]] = []
        missing: list[str] = []
        contradictions: list[str] = []

        for attr_name, target_val in target.known_attributes.items():
            obj_val = obj.get_attribute_value(attr_name)
            if obj_val is None:
                missing.append(attr_name)
            elif not _semantic_match(obj_val, target_val):
                contradictions.append(f"{attr_name}: obj={obj_val!r} vs target={target_val!r}")
            else:
                matched.append({"attribute": attr_name, "value": obj_val})

        for attr_name, neg_val in target.negative_attributes.items():
            obj_val = obj.get_attribute_value(attr_name)
            if obj_val is not None and _semantic_match(obj_val, neg_val):
                contradictions.append(f"{attr_name}: obj={obj_val!r} matches forbidden {neg_val!r}")

        score = GraphMatcher.compute_alignment(obj, target)
        return {
            "score": score,
            "matched": matched,
            "missing_attributes": missing,
            "contradictions": contradictions,
            "target_known": dict(target.known_attributes),
            "target_negative": dict(target.negative_attributes),
        }

    @staticmethod
    def find_contradictions(obj: ObjectNode, target: TargetFacts) -> list[str]:
        contradictions = []
        for attr_name, target_val in target.known_attributes.items():
            obj_val = obj.get_attribute_value(attr_name)
            if obj_val is not None and not _semantic_match(obj_val, target_val):
                contradictions.append(f"{attr_name}: obj={obj_val}, target={target_val}")
        for attr_name, neg_val in target.negative_attributes.items():
            obj_val = obj.get_attribute_value(attr_name)
            if obj_val is not None and _semantic_match(obj_val, neg_val):
                contradictions.append(f"{attr_name}: obj={obj_val}, target NOT {neg_val}")
        return contradictions

    @staticmethod
    def compute_discriminative_power(attr_name: str, instances: list[ObjectNode]) -> float:
        if len(instances) <= 1:
            return 0.0
        values = []
        for inst in instances:
            val = inst.get_attribute_value(attr_name)
            if val is not None:
                values.append(val.lower())
        if not values:
            return 0.5
        unique = set(values)
        if len(unique) == 1:
            return 0.0
        counts = Counter(values)
        total = len(values)
        entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
        max_ent = math.log2(len(unique))
        return entropy / max_ent if max_ent > 0 else 0.0
