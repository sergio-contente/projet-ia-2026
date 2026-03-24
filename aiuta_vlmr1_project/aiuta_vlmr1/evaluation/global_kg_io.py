"""Load/save global KG JSON (legacy pure graph or bundle with sample_id ↔ hash map)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..knowledge_graph.scene_graph import SceneKnowledgeGraph

GLOBAL_KG_BUNDLE_FORMAT_V1 = "global_kg_bundle_v1"


def save_global_kg_bundle(
    kg: SceneKnowledgeGraph,
    sample_id_to_image_hash: dict[str, str],
    path: str | Path,
) -> None:
    """Write bundle: graph + ``sample_id_to_image_hash`` for lookup / provenance."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format": GLOBAL_KG_BUNDLE_FORMAT_V1,
        "graph": kg.to_dict(),
        "sample_id_to_image_hash": dict(sample_id_to_image_hash),
    }
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def load_global_kg_for_eval(path: str | Path) -> tuple[SceneKnowledgeGraph, dict[str, str]]:
    """
    Load a global KG file.

    - **Bundle** (``format == global_kg_bundle_v1``): returns graph + mapping.
    - **Legacy** (plain :meth:`SceneKnowledgeGraph.to_dict` output): mapping is ``{}``.
    """
    p = Path(path)
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and data.get("format") == GLOBAL_KG_BUNDLE_FORMAT_V1:
        graph = data.get("graph")
        if not isinstance(graph, dict):
            raise ValueError("bundle missing 'graph' dict")
        mapping = dict(data.get("sample_id_to_image_hash") or {})
        return SceneKnowledgeGraph.from_dict(graph), mapping
    if not isinstance(data, dict):
        raise ValueError("global KG JSON must be an object")
    return SceneKnowledgeGraph.from_dict(data), {}
