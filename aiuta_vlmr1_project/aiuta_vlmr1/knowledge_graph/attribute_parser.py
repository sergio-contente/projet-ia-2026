"""attribute_parser.py — Parse structured JSON attributes from VLM <answer> tags."""
from __future__ import annotations

import json
import re
from typing import Any

from .schema import Attribute, AttributeSource, Certainty
from .triple_extractor import TripleExtractor

# Fields we expect in the attribute JSON.
ATTRIBUTE_FIELDS = ("color", "material", "size", "features", "location", "pattern")


def _extract_answer_json(text: str) -> str | None:
    """Extract content between <answer> tags, stripping whitespace."""
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def _try_parse_json(raw: str) -> dict[str, Any] | None:
    """Best-effort JSON parse with common VLM quirks (trailing commas, etc.)."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Strip trailing commas before closing brace
    cleaned = re.sub(r",\s*}", "}", raw)
    cleaned = re.sub(r",\s*]", "]", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Try to extract the first {...} block
    m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def parse_attribute_json(
    text: str,
    category: str = "object",
    timestep: int = 0,
) -> list[Attribute]:
    """
    Parse the attribute JSON from a VLM response.

    Returns a list of ``Attribute`` objects for each non-null field.
    Falls back to ``TripleExtractor`` if JSON parsing fails entirely.
    """
    answer_str = _extract_answer_json(text)
    if answer_str is None:
        # No <answer> tags — try the whole text as JSON
        answer_str = text

    data = _try_parse_json(answer_str)
    if data is None:
        # Fallback: use TripleExtractor on the raw text
        return TripleExtractor.extract_attributes(text, timestep=timestep)

    attributes: list[Attribute] = []
    for field_name in ATTRIBUTE_FIELDS:
        value = data.get(field_name)
        if value is None or (isinstance(value, str) and value.strip().lower() in ("null", "none", "")):
            continue
        attributes.append(Attribute(
            name=field_name,
            value=str(value).strip(),
            certainty=Certainty.HIGH,
            source=AttributeSource.VLM_REASONING,
            timestep=timestep,
        ))

    return attributes
