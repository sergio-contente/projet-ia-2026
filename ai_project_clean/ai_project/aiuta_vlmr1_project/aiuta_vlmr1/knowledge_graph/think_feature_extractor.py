"""
think_feature_extractor.py — Extract discriminative visual features from
VLM structured description output for contextual question generation.

Features are decomposed into (qualifier, subject, type) so that open-ended
questions can be generated instead of yes/no (which the VLM-R1 3B oracle
answers with IDK ~70% of the time).

Example: "green mattress" → qualifier="green", subject="mattress", type="color"
  → question: "What color is the mattress on the bed?"
  → obj attr: think_mattress = "green"
  → oracle answers: "white"
  → target attr: think_mattress = "white"
  → alignment: "green" ≠ "white" → mismatch → CONTINUE
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

_QUALIFIER_TO_QUESTION_TYPE: dict[str, str] = {
    "red": "color", "blue": "color", "green": "color", "black": "color",
    "white": "color", "yellow": "color", "brown": "color", "gray": "color",
    "grey": "color", "orange": "color", "pink": "color", "purple": "color",
    "beige": "color", "dark": "color", "light": "color",
    "dark-colored": "color", "light-colored": "color",
    "wooden": "material", "metal": "material", "leather": "material",
    "glass": "material", "fabric": "material", "ceramic": "material",
    "marble": "material", "wicker": "material", "plastic": "material",
    "cotton": "material", "silk": "material", "wool": "material",
    "large": "size", "small": "size", "big": "size", "tiny": "size",
    "medium": "size", "tall": "size", "short": "size",
    "striped": "pattern", "checkered": "pattern", "plaid": "pattern",
    "floral": "pattern", "patterned": "pattern", "plain": "pattern",
    "smooth": "texture", "rough": "texture", "textured": "texture",
    "quilted": "texture", "embroidered": "texture", "soft": "texture",
}

_QUESTION_TEMPLATES_BY_TYPE: dict[str, str] = {
    "color": "What color is the {subject} on the {category}?",
    "material": "What material is the {subject} on the {category} made of?",
    "size": "Is the {subject} on the {category} large or small?",
    "pattern": "What pattern does the {subject} on the {category} have?",
    "texture": "What texture does the {subject} on the {category} have?",
}


def extract_think_features(
    description: str, category: str, max_features: int = 5
) -> list[str]:
    """
    Extract distinctive visual features from a VLM structured description.

    Expects comma-separated or newline-separated short phrases like:
      "dark-colored mattress, textured surface, fabric material, against wall"
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
                feat = feat[len(prefix):]
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


def _decompose_feature(feature: str) -> tuple[str, str, str]:
    """
    Decompose "green mattress" → ("green", "mattress", "color").
    Decompose "near window"    → ("window", "", "spatial").
    """
    words = feature.lower().strip().split()

    if words[0] in ("near", "against", "beside", "behind", "next"):
        subject = " ".join(words[1:]).lstrip("to ")
        return (subject, "", "spatial")

    if len(words) >= 2:
        for i in range(1, len(words)):
            if words[0] in _QUALIFIER_TO_QUESTION_TYPE:
                qualifier = " ".join(words[:i])
                subject = " ".join(words[i:])
                q_type = _QUALIFIER_TO_QUESTION_TYPE[words[0]]
                return (qualifier, subject, q_type)
        return (words[0], " ".join(words[1:]), "unknown")

    return (feature, "", "unknown")


def feature_to_question(feature: str, category: str) -> str:
    """
    Convert a visual feature into an OPEN-ENDED question for the oracle.

    "green mattress" → "What color is the mattress on the bed?"
    "metal frame"    → "What material is the frame on the bed made of?"
    "near window"    → "What is the bed near or next to?"
    """
    qualifier, subject, q_type = _decompose_feature(feature)

    if q_type == "spatial":
        return f"What is the {category} near or next to?"

    if q_type in _QUESTION_TEMPLATES_BY_TYPE and subject:
        return _QUESTION_TEMPLATES_BY_TYPE[q_type].format(
            subject=subject, category=category
        )

    if subject:
        return f"Describe the {subject} on the {category}."

    return f"Does the {category} have a {feature}?"


def feature_to_attribute_name(feature: str) -> str:
    """
    Noun-based KG attribute name.

    "green mattress" → "think_mattress"
    "metal frame"    → "think_frame"
    "near window"    → "think_near"
    """
    _qualifier, subject, q_type = _decompose_feature(feature)
    if q_type == "spatial":
        base = "near"
    elif subject:
        base = re.sub(r"[^a-z0-9]+", "_", subject).strip("_")
    else:
        base = re.sub(r"[^a-z0-9]+", "_", feature.lower()).strip("_")
    if len(base) > 30:
        base = base[:30].rstrip("_")
    return f"think_{base}"


def feature_to_qualifier(feature: str) -> str:
    """
    Extract the qualifier (adjective/value) that the detection observed.

    "green mattress" → "green"
    "metal frame"    → "metal"
    "near window"    → "window"
    """
    qualifier, _subject, _q_type = _decompose_feature(feature)
    return qualifier
