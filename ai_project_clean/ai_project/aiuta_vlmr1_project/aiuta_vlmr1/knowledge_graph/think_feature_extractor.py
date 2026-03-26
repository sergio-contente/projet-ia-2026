"""
think_feature_extractor.py — Extract discriminative visual features from
VLM structured description output for contextual question generation.

Uses comma-separated VLM output instead of regex pattern matching.
Cost: 0 extra parsing — features come from the description pass split.
"""
from __future__ import annotations

import re

_GENERIC_FEATURES = frozenset({
    "bed", "a bed", "the bed", "chair", "a chair", "the chair",
    "cabinet", "a cabinet", "the cabinet", "couch", "a couch",
    "picture", "a picture", "the object", "an object",
    "the room", "the image", "the scene", "the wall", "the floor",
    "the ceiling", "this image", "the background",
    "unknown", "none", "n/a", "not visible", "not sure",
})

_META_PREFIXES = (
    "i can", "i see", "the image", "this is", "there is", "it appears",
    "it looks", "it seems", "the photo", "in this",
)


def extract_think_features(
    description: str, category: str, max_features: int = 5
) -> list[str]:
    """
    Extract distinctive visual features from a VLM structured description.

    Expects comma-separated or newline-separated short phrases like:
      "dark-colored mattress, textured surface, fabric material, against wall"

    Returns cleaned list: ["dark-colored mattress", "textured surface", ...]
    """
    if not description:
        return []

    raw_parts = re.split(r"[,\n]|\band\b", description)

    features: list[str] = []
    seen: set[str] = set()

    for part in raw_parts:
        feat = part.strip().rstrip(".,;:").lstrip("-*\u2022123456789. ")
        feat = feat.strip()
        for prefix in ("a ", "an ", "the ", "its ", "this ", "it has ", "with "):
            if feat.lower().startswith(prefix):
                feat = feat[len(prefix) :]
        feat = feat.strip()

        low = feat.lower()
        if len(feat) < 3:
            continue
        if low in _GENERIC_FEATURES:
            continue
        if low == category.lower():
            continue
        if len(feat.split()) > 6:
            continue
        if any(low.startswith(p) for p in _META_PREFIXES):
            continue

        if low not in seen:
            seen.add(low)
            features.append(feat)

        if len(features) >= max_features:
            break

    return features


def feature_to_question(feature: str, category: str) -> str:
    """Convert a visual feature into a yes/no question for the oracle."""
    low = feature.lower()
    if any(low.startswith(p) for p in ("near ", "next to ", "beside ", "against ", "in front of ")):
        return f"Is the {category} {feature}?"
    return f"Does the {category} have a {feature}?"


def feature_to_attribute_name(feature: str) -> str:
    """Convert feature phrase to a KG attribute name (think_*)."""
    clean = re.sub(r"[^a-z0-9]+", "_", feature.lower()).strip("_")
    if len(clean) > 40:
        clean = clean[:40].rstrip("_")
    return f"think_{clean}"
