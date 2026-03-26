"""
think_feature_extractor.py — Extract discriminative visual features from
VLM-R1 <think> blocks for contextual question generation.

Replaces generic template questions ("What color?") with specific feature
questions ("Does the bed have a blue mattress?") that better discriminate
between object instances.

Cost: 0 extra VLM calls — features come from the existing detection reasoning.
"""
from __future__ import annotations

import re

_FEATURE_PATTERNS = [
    # "with a blue mattress" / "with wooden frame"
    r"(?:with|has|have)\s+(?:a\s+|an\s+)?((?:\w+\s+){1,3}\w+?)(?:[.,;]|\s+and\s|\s+near|\s+in\s|$)",
    # "{adjective} {noun}" combos for visual descriptors
    r"((?:(?:blue|red|white|black|green|yellow|brown|gray|grey|pink|purple|orange|beige|dark|light|"
    r"striped|checkered|plaid|floral|patterned|wooden|metal|leather|glass|fabric|quilted|"
    r"large|small|round|square|tall|short|thin|thick|old|new|modern|antique|rustic)\s+)+"
    r"(?:mattress|pillow|blanket|sheet|frame|headboard|footboard|cover|cushion|"
    r"drawer|handle|door|shelf|leg|arm|seat|back|top|surface|panel|"
    r"lamp|table|nightstand|rug|curtain|decoration|pattern|design)\w*)",
]

_GENERIC_FEATURES = frozenset({
    "the bed", "a bed", "the object", "an object", "the room",
    "the image", "the scene", "the wall", "the floor",
})


def extract_think_features(
    reasoning: str, category: str, max_features: int = 5
) -> list[str]:
    """
    Extract distinctive visual feature phrases from a <think> reasoning block.

    Returns short descriptive phrases like:
      ["blue mattress", "wooden frame", "striped blanket"]
    """
    if not reasoning:
        return []

    text = reasoning.lower().strip()
    features: list[str] = []
    seen: set[str] = set()

    for pattern in _FEATURE_PATTERNS:
        for m in re.finditer(pattern, text):
            feat = m.group(1).strip().rstrip(".,;")
            for prefix in ("a ", "an ", "the ", "its ", "this "):
                if feat.startswith(prefix):
                    feat = feat[len(prefix):]
            feat = feat.strip()
            if len(feat) < 4 or len(feat.split()) < 2:
                continue
            if feat in seen or feat in _GENERIC_FEATURES:
                continue
            if feat == category.lower() or feat == f"the {category.lower()}":
                continue
            seen.add(feat)
            features.append(feat)
            if len(features) >= max_features:
                break
        if len(features) >= max_features:
            break

    return features


def feature_to_question(feature: str, category: str) -> str:
    """Convert a visual feature into a yes/no question for the oracle."""
    return f"Does the {category} have a {feature}?"


def feature_to_attribute_name(feature: str) -> str:
    """Convert feature phrase to a KG attribute name (think_*)."""
    clean = re.sub(r"[^a-z0-9]+", "_", feature.lower()).strip("_")
    return f"think_{clean}"
