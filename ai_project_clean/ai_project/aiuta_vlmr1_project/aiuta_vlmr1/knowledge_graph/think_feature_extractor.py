"""
think_feature_extractor.py — Extract discriminative visual features from
VLM description passes for contextual question generation.

Replaces generic template questions ("What color?") with specific feature
questions ("Does the bed have a blue mattress?") that better discriminate
between object instances.

Features come from a single description-pass VLM call (+1 call per detection).
"""
from __future__ import annotations

import re

_FEATURE_PATTERNS = [
    # "has/with a {adj} {noun}"
    r"(?:has|with|have)\s+(?:a\s+|an\s+)?((?:(?:\w+\s+){0,2})"
    r"(?:mattress|pillow|blanket|sheet|frame|headboard|footboard|cover|cushion|"
    r"drawer|handle|door|shelf|leg|arm|seat|back|surface|panel|"
    r"lamp|table|nightstand|rug|curtain|pattern|design|"
    r"cabinet|countertop|sink|faucet|mirror|towel|"
    r"armrest|backrest|upholstery|finish|trim|base|"
    r"picture|painting|poster|photograph|artwork)\w*)",
    # "{color/adj} {noun}" standalone
    r"\b((?:blue|red|white|black|green|yellow|brown|gray|grey|pink|purple|orange|beige|"
    r"dark|light|striped|checkered|plaid|floral|patterned|quilted|embroidered|"
    r"wooden|metal|leather|glass|fabric|ceramic|marble|wicker|"
    r"large|small|round|square|tall|short|modern|antique|rustic|vintage)\s+"
    r"(?:mattress|pillow|blanket|sheet|frame|headboard|cover|cushion|"
    r"drawer|handle|door|shelf|surface|panel|"
    r"lamp|table|nightstand|rug|curtain|"
    r"cabinet|countertop|sink|faucet|"
    r"armrest|backrest|upholstery|finish|base|"
    r"picture|painting|poster|photograph)\w*)",
    # Spatial: "near/next to {object}"
    r"(?:near|next to|beside|against|in front of)\s+(?:a\s+|an\s+|the\s+)?"
    r"((?:\w+\s+){0,1}(?:window|wall|door|desk|table|chair|nightstand|lamp|"
    r"dresser|closet|mirror|shelf|bookcase|couch|sofa|sink|toilet|bathtub|"
    r"stove|refrigerator|counter)\w*)",
]

_GENERIC_FEATURES = frozenset({
    "the bed", "a bed", "the object", "an object", "the room",
    "the image", "the scene", "the wall", "the floor",
    "the ceiling", "this image", "the background",
    "the chair", "a chair", "the couch", "a couch",
    "the cabinet", "a cabinet", "the picture", "a picture",
})


def extract_think_features(
    reasoning: str, category: str, max_features: int = 5
) -> list[str]:
    """
    Extract distinctive visual feature phrases from a description or reasoning block.

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
