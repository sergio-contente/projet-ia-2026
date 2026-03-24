"""
IDKVQA-specific KG helpers: question parsing, attribute lookup, hybrid KG+VQA fusion.

Used by the primary offline benchmark (``run_idkvqa_benchmark``) for ``mode="kg"``
and ``mode="kg_threshold"``.
"""
from __future__ import annotations

import re
from typing import Any

from ..knowledge_graph.triple_extractor import TripleExtractor
from .answer_normalization import LABEL_IDK, LABEL_NO, LABEL_YES, normalize_yes_no_idk

ATTRIBUTE_PATTERNS: list[tuple[str, str]] = [
    (r"(?:is|are).*\b(red|blue|green|black|white|yellow|brown|gray|grey|orange|pink|purple|dark|light|beige)\b", "color"),
    (r"made of\s+(\w+)", "material"),
    (r"(?:is|are).*\b(wood|wooden|metal|metallic|plastic|glass|leather|fabric|ceramic|stone|marble|porcelain|laminate)\b", "material"),
    (r"(?:have|has)\s+(?:a\s+)?glass\s+door", "has_glass_door"),
    (r"(?:have|has)\s+(?:a\s+)?handle", "has_handle"),
    (r"(?:have|has)\s+(?:a\s+)?drawers?", "has_drawer"),
    (r"(?:is|are).*\bopen\b", "is_open"),
    (r"(?:is|are).*\bclosed\b", "is_open"),
    (r"(?:is|are).*\bwall.?mounted\b", "is_wall_mounted"),
    (r"(?:is|are).*\btufted\b", "has_tufted"),
    (r"(?:is|are).*\b(modern|antique|minimalist|traditional)\b", "style"),
    (r"(?:is|are).*\b(large|small|big|tiny|compact|tall|short)\b", "size"),
    (r"(?:is|are).*(?:in|located\s+in)\s+(?:a\s+|the\s+)?([\w\s]+?)\s*\?", "location"),
    (r"(?:is|are).*(?:against|near|next\s+to|beside)\s+(?:a\s+|the\s+)?(\w+)", "spatial"),
    (r"(?:is|are).*\bcorner\b", "in_corner"),
    (r"(?:is|are).*\bwall\b", "near_wall"),
    (r"(?:is\s+there|are\s+there)\s+(?:a\s+|any\s+)?(\w+)", "nearby_object"),
]


_SUBJECT_SKIP_WORDS = {
    "it", "this", "that", "there", "here", "image", "picture", "scene", "room", "true", "same",
    "possible", "correct", "visible", "present", "something", "anything",
    "red", "blue", "green", "black", "white", "yellow", "brown", "gray", "grey", "orange", "pink",
    "dark", "light", "beige", "purple",
    "large", "small", "big", "tiny", "compact", "tall", "short",
}


def subject_category_from_idkvqa_question(question: str) -> str:
    """
    Heuristic **object category** (subject noun) from an IDKVQA question.

    Examples: "Is the chair red?" → ``chair``; "Is the red chair wooden?" → ``chair``.
    Falls back to ``object`` when parsing fails.
    """
    low = question.lower().strip()
    m = re.search(r"(?:is\s+there|are\s+there)\s+(?:a|an|any|some)\s+([a-z][a-z0-9\-]*)", low)
    if m:
        w = m.group(1)
        if w not in _SUBJECT_SKIP_WORDS:
            return w
    m = re.search(
        r"\b(?:is|are|does|do)\s+the\s+([a-z][a-z0-9]*)(?:\s+([a-z][a-z0-9]*))?",
        low,
    )
    if m:
        w1, w2 = m.group(1), m.group(2)
        if w1 not in _SUBJECT_SKIP_WORDS:
            return w1
        if w2 and w2 not in _SUBJECT_SKIP_WORDS:
            return w2
    m = re.search(r"\b(?:is|are)\s+(?:a|an)\s+([a-z][a-z0-9\-]*)", low)
    if m:
        w = m.group(1)
        if w not in _SUBJECT_SKIP_WORDS:
            return w
    return "object"


def parse_question_attribute(question: str) -> tuple[str, str | None]:
    """Best-effort attribute key and optional literal value from the question text."""
    q = question.lower().strip()
    for pattern, attr_type in ATTRIBUTE_PATTERNS:
        match = re.search(pattern, q)
        if match:
            value = match.group(1) if match.lastindex else None
            return attr_type, value
    return "unknown", None


def coarse_question_taxonomy(question: str) -> str:
    """
    Coarse type for paper breakdowns. **Heuristic only** — not from dataset metadata.

    Groups: existence, attribute, color, material, location, spatial, size, count, style, unknown_other.
    """
    q = question.lower()
    if re.search(r"\bhow\s+many\b", q) or re.search(r"\bnumber\s+of\b", q):
        return "count"
    if any(w in q for w in ["is there", "are there", "is it true that there"]):
        return "existence"
    if any(w in q for w in ["in the", "in a", "located", "which room", "kitchen", "bedroom", "bathroom"]):
        if "color" not in q and "made" not in q:
            return "location"
    if any(w in q for w in ["color", "red", "blue", "green", "black", "white", "brown", "gray", "grey", "dark", "light", "beige"]):
        return "color"
    if any(w in q for w in ["made of", "material", "wood", "metal", "glass", "leather", "fabric", "ceramic", "porcelain"]):
        return "material"
    if any(w in q for w in ["near", "next to", "corner", "against", "behind", "beside", "spatial"]):
        return "spatial"
    if any(w in q for w in ["large", "small", "big", "tiny", "compact", "tall", "size"]):
        return "size"
    if any(w in q for w in ["have", "has", "does", "drawer", "handle", "door", "open", "closed"]):
        return "attribute"
    if any(w in q for w in ["style", "modern", "antique", "minimalist", "traditional", "cozy", "warm"]):
        return "style"
    return "unknown_other"


def classify_question_type(question: str) -> str:
    """Alias for :func:`coarse_question_taxonomy` (backward compatible name)."""
    return coarse_question_taxonomy(question)


def kg_answer_from_attributes(
    kg_attributes: dict[str, str],
    attr_type: str,
    attr_value: str | None,
) -> str:
    """Map structured KG attributes to Yes / No / I don't know for the parsed question slot."""
    kg_val = kg_attributes.get(attr_type)
    if kg_val is None:
        return LABEL_IDK
    if attr_value is not None:
        return LABEL_YES if kg_val.lower() == attr_value.lower() else LABEL_NO
    lv = kg_val.lower()
    if lv in ("true", "yes"):
        return LABEL_YES
    if lv in ("false", "no"):
        return LABEL_NO
    return LABEL_YES


def compute_kg_entropy_modulation(
    raw_predicted: str,
    kg_attributes: dict[str, str],
    attr_type: str,
    attr_value: str | None,
    entropy: float | None,
    *,
    boost_factor: float = 0.5,
    penalty_factor: float = 2.0,
    cap: float = 0.30,
) -> tuple[str, float | None, str]:
    """
    Modulate VLM entropy using KG evidence. KG NEVER changes the answer text.

    Returns ``(final_answer, adjusted_entropy, kg_signal)`` where ``kg_signal`` is one of
    ``agree``, ``contradict``, ``no_info``.
    """
    rp = normalize_yes_no_idk(raw_predicted)
    if entropy is None:
        return raw_predicted, entropy, "no_info"
    if rp == LABEL_IDK:
        return raw_predicted, entropy, "no_info"

    kg_answer = kg_answer_from_attributes(kg_attributes, attr_type, attr_value)
    if kg_answer == LABEL_IDK:
        return raw_predicted, entropy, "no_info"
    if kg_answer == rp:
        adjusted = float(entropy) * float(boost_factor)
        signal = "agree"
    else:
        adjusted = float(entropy) * float(penalty_factor)
        signal = "contradict"

    adjusted = max(0.0, min(adjusted, float(cap)))
    return raw_predicted, adjusted, signal


def compute_dual_kg_entropy_modulation(
    raw_predicted: str,
    global_kg_attributes: dict[str, str],
    episode_kg_attributes: dict[str, str],
    attr_type: str,
    attr_value: str | None,
    entropy: float | None,
    *,
    boost_factor: float = 0.5,
    penalty_factor: float = 2.0,
    cap: float = 0.30,
) -> tuple[str, float | None, dict[str, str]]:
    """
    Dual KG entropy modulation (global + episodic). Answer text unchanged; entropy adjusted twice.
    """
    rp = normalize_yes_no_idk(raw_predicted)
    if entropy is None or rp == LABEL_IDK:
        return raw_predicted, entropy, {"global": "no_info", "episode": "no_info"}

    _, e1, sig_global = compute_kg_entropy_modulation(
        raw_predicted,
        global_kg_attributes,
        attr_type,
        attr_value,
        entropy,
        boost_factor=boost_factor,
        penalty_factor=penalty_factor,
        cap=1.0,
    )
    _, e2, sig_episode = compute_kg_entropy_modulation(
        raw_predicted,
        episode_kg_attributes,
        attr_type,
        attr_value,
        e1,
        boost_factor=boost_factor,
        penalty_factor=penalty_factor,
        cap=cap,
    )
    return raw_predicted, e2, {"global": sig_global, "episode": sig_episode}


def build_kg_attributes_from_detection(
    detection_reasoning: str,
    category: str = "object",
    queried_objects: list[str] | None = None,
    timestep: int = 0,
) -> tuple[dict[str, str], Any]:
    """Run ``TripleExtractor`` on detection reasoning and flatten to a string dict."""
    if queried_objects is None:
        queried_objects = []
    extraction = TripleExtractor.extract_all(
        reasoning=detection_reasoning,
        category=category,
        queried_objects=queried_objects,
        timestep=timestep,
    )
    kg_attributes: dict[str, str] = {a.name: a.value for a in extraction.attributes}
    for rel in extraction.spatial_relations:
        kg_attributes[rel.relation] = rel.reference
    return kg_attributes, extraction


def enrich_kg_from_reasoning(
    kg_broad: dict[str, str],
    detection_reasoning: str,
    attr_type: str,
) -> None:
    """Fill missing color/material hints from free text (in-place)."""
    low = detection_reasoning.lower()
    if attr_type == "color" and "color" not in kg_broad:
        for c in ["red", "blue", "green", "black", "white", "yellow", "brown", "gray", "grey", "orange", "pink", "dark", "light"]:
            if c in low:
                kg_broad["color"] = c
                break
    if attr_type == "material" and "material" not in kg_broad:
        for m in ["wood", "wooden", "metal", "glass", "leather", "fabric", "ceramic", "plastic", "stone"]:
            if m in low:
                kg_broad["material"] = m
                break


def compute_kg_hybrid_prediction(
    raw_predicted: str,
    vqa_reasoning: str,
    kg_attributes: dict[str, str],
    attr_type: str,
    attr_value: str | None,
    detection_reasoning: str,
) -> str:
    """
    Combine raw VQA with KG medium path and hedging heuristics (same intent as legacy kg_vs_raw).
    All inputs should use canonical labels where possible; ``raw_predicted`` is normalized first.
    rp = normalize_yes_no_idk(raw_answer) internally expected by caller.
    """
    kg_broad = dict(kg_attributes)
    enrich_kg_from_reasoning(kg_broad, detection_reasoning, attr_type)
    kg_medium = kg_answer_from_attributes(kg_broad, attr_type, attr_value)

    hedging = [
        "appears", "seems", "might", "possibly", "likely", "probably", "could be",
        "hard to tell", "not sure", "unclear", "cannot determine",
    ]
    has_hedging = any(h in vqa_reasoning.lower() for h in hedging)

    if has_hedging and attr_type not in kg_broad:
        return LABEL_IDK
    if not has_hedging and attr_type in kg_broad:
        return raw_predicted
    if attr_type in kg_broad:
        return kg_medium
    return LABEL_IDK


def compute_kg_hybrid_prediction_relaxed(
    raw_predicted: str,
    vqa_reasoning: str,
    kg_attributes: dict[str, str],
    attr_type: str,
    attr_value: str | None,
    detection_reasoning: str,
) -> str:
    """
    Same fusion as :func:`compute_kg_hybrid_prediction`, except when there is no KG evidence
    for ``attr_type`` and the VQA text is not hedging: return ``raw_predicted`` instead of
    ``LABEL_IDK`` (trust the VLM when it appears confident).
    """
    kg_broad = dict(kg_attributes)
    enrich_kg_from_reasoning(kg_broad, detection_reasoning, attr_type)
    kg_medium = kg_answer_from_attributes(kg_broad, attr_type, attr_value)

    hedging = [
        "appears", "seems", "might", "possibly", "likely", "probably", "could be",
        "hard to tell", "not sure", "unclear", "cannot determine",
    ]
    has_hedging = any(h in vqa_reasoning.lower() for h in hedging)

    if has_hedging and attr_type not in kg_broad:
        return LABEL_IDK
    if not has_hedging and attr_type in kg_broad:
        return raw_predicted
    if attr_type in kg_broad:
        return kg_medium
    return raw_predicted


def compute_kg_hybrid_prediction_entropy(
    raw_predicted: str,
    vqa_reasoning: str,
    kg_attributes: dict[str, str],
    attr_type: str,
    attr_value: str | None,
    detection_reasoning: str,
    entropy: float | None,
    entropy_tau: float = 0.09,
) -> str:
    """
    Same fusion as :func:`compute_kg_hybrid_prediction_relaxed`, but fallback to raw VLM
    only when token entropy is below ``entropy_tau``.
    """
    kg_broad = dict(kg_attributes)
    enrich_kg_from_reasoning(kg_broad, detection_reasoning, attr_type)
    kg_medium = kg_answer_from_attributes(kg_broad, attr_type, attr_value)

    hedging = [
        "appears", "seems", "might", "possibly", "likely", "probably", "could be",
        "hard to tell", "not sure", "unclear", "cannot determine",
    ]
    has_hedging = any(h in vqa_reasoning.lower() for h in hedging)

    if has_hedging and attr_type not in kg_broad:
        return LABEL_IDK
    if not has_hedging and attr_type in kg_broad:
        return raw_predicted
    if attr_type in kg_broad:
        return kg_medium
    if entropy is not None and entropy < entropy_tau:
        return raw_predicted
    return LABEL_IDK


def normalize_prediction_label(text: str) -> str:
    """Normalize model output to Yes / No / I don't know when possible."""
    return normalize_yes_no_idk(text)
