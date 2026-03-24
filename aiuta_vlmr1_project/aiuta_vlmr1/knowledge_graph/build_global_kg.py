"""
build_global_kg — Offline **global** knowledge graph for IDKVQA.

- **Dedup:** one **OVD** VLM call per unique image (SHA-256 of PNG bytes).
- **Per-object:** :class:`VLMr1Detector` returns ``Detection`` entries (label, bbox, …);
  triples are extracted per detection with :class:`TripleExtractor` on text focused on
  that label (see :func:`_reasoning_slice_for_label`).
- **Storage:** one ``ObjectNode`` per ``sample_id``; ``image_id`` matches eval ``sample_id``.
- **Bundle:** :func:`aiuta_vlmr1.evaluation.global_kg_io.save_global_kg_bundle`.

**Oracle / leakage note:** building on the same split you evaluate is an upper-bound-style
analysis; document accordingly in papers.
"""
from __future__ import annotations

import argparse
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Config
from ..detector.base import Detection
from ..detector.vlmr1_detector import VLMr1Detector
from ..evaluation.global_kg_io import save_global_kg_bundle
from ..evaluation.idkvqa_eval import stable_idkvqa_image_id
from ..evaluation.idkvqa_kg import enrich_kg_from_reasoning, subject_category_from_idkvqa_question
from ..knowledge_graph.schema import Attribute, AttributeSource, Certainty
from ..utils.model_loader import ModelLoader
from .scene_graph import SceneKnowledgeGraph
from .triple_extractor import TripleExtractor


def _reasoning_slice_for_label(global_reasoning: str | None, label: str) -> str:
    """
    Prefer sentences that mention ``label`` so :class:`TripleExtractor` sees object-local text.

    The VLM parser often attaches one global ``reasoning_text`` to every box; this slice
    approximates per-object context without changing the detector.
    """
    if not global_reasoning:
        return ""
    lbl = label.strip().lower()
    if not lbl:
        return global_reasoning.strip()
    low = global_reasoning.lower()
    if lbl not in low:
        return global_reasoning.strip()
    parts = re.split(r"(?<=[.!?])\s+", global_reasoning.strip())
    hits = [p for p in parts if lbl in p.lower()]
    if hits:
        return " ".join(hits)
    return global_reasoning.strip()


def _pick_detection_for_question(detections: list[Detection], q_cat: str) -> Detection | None:
    """Map question subject category to one :class:`Detection` (first match)."""
    if not detections:
        return None
    qc = (q_cat or "object").strip().lower()
    if qc == "object":
        return detections[0]
    for d in detections:
        dl = d.label.strip().lower()
        if dl == qc:
            return d
    for d in detections:
        dl = d.label.strip().lower()
        if qc in dl or dl in qc:
            return d
    return None


def _attrs_and_spatial_from_extraction(
    extraction: Any,
    reasoning_for_enrich: str,
    *,
    certainty: Certainty = Certainty.MEDIUM,
) -> tuple[list[Attribute], list[SpatialRelation]]:
    kg_broad: dict[str, str] = {}
    for a in extraction.attributes:
        kg_broad[a.name] = a.value
    for at in ("color", "material"):
        enrich_kg_from_reasoning(kg_broad, reasoning_for_enrich, at)
    attrs = [
        Attribute(
            name=k,
            value=str(v),
            certainty=certainty,
            source=AttributeSource.VLM_REASONING,
            timestep=0,
        )
        for k, v in kg_broad.items()
    ]
    return attrs, list(extraction.spatial_relations)


def _pil_to_observation(pil_image: Any) -> np.ndarray:
    arr = np.asarray(pil_image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr.astype(np.uint8, copy=False)


def _load_hf_rows(split: str, seed: int | None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    rows: list[dict[str, Any]] = list(load_dataset("ftaioli/IDKVQA", split=split))
    if seed is not None:
        random.Random(seed).shuffle(rows)
    return rows


def build_global_kg(
    config: Config,
    split: str = "val",
    limit: int | None = None,
    seed: int = 42,
    output_path: str = "results/global_kg.json",
) -> SceneKnowledgeGraph:
    """
    Build and persist a global KG.

    Parameters
    ----------
    limit :
        Maximum number of **unique images** to run OVD on (after shuffling group order).
    """
    random.seed(seed)
    rows = _load_hf_rows(split, seed)
    sample_id_to_hash: dict[str, str] = {}

    groups: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for global_idx, row in enumerate(rows):
        pil = row["image"]
        img_hash = stable_idkvqa_image_id(pil)
        sid = str(row.get("id", global_idx))
        sample_id_to_hash[sid] = img_hash
        groups[img_hash].append((sid, row))

    ordered_hashes = list(groups.keys())
    random.Random(seed).shuffle(ordered_hashes)

    ModelLoader.get_instance(config.model)
    detector = VLMr1Detector(config)
    kg = SceneKnowledgeGraph()
    n_det = 0

    for img_hash in ordered_hashes:
        if limit is not None and n_det >= limit:
            break
        members = groups[img_hash]
        row0 = members[0][1]
        pil_image = row0["image"]
        obs = _pil_to_observation(pil_image)

        image_categories: set[str] = set()
        for _sid, row in members:
            cat = subject_category_from_idkvqa_question(str(row.get("question", "")))
            if cat != "object":
                image_categories.add(cat.lower())
        if not image_categories:
            image_categories = {"object"}
        cats_list = sorted(image_categories)

        det_result = detector.detect_from_observation(obs, cats_list, kg_context=None)
        dets = det_result.detections
        scene_reasoning = (det_result.reasoning_text or "").strip()

        per_det_extractions: list[tuple[Detection, Any]] = []
        for det in dets:
            slice_text = _reasoning_slice_for_label(det_result.reasoning_text, det.label)
            text = slice_text or scene_reasoning
            ext = TripleExtractor.extract_all(
                reasoning=text,
                category=det.label,
                queried_objects=[],
                timestep=0,
            )
            per_det_extractions.append((det, ext))

        for sid, row in members:
            q_cat = subject_category_from_idkvqa_question(str(row.get("question", "")))
            det = _pick_detection_for_question(dets, q_cat)

            if det is not None:
                di = dets.index(det)
                _, extraction = per_det_extractions[di]
                enrich_text = (det.reasoning or scene_reasoning or "")
                attrs, spatial_rels = _attrs_and_spatial_from_extraction(
                    extraction,
                    enrich_text,
                    certainty=Certainty.MEDIUM,
                )
                node = kg.add_object_merged(
                    det.label,
                    bbox=det.bbox if det.bbox and len(det.bbox) == 4 else None,
                    timestep=0,
                    image_id=sid,
                )
                if attrs:
                    kg.update_attributes(node.obj_id, attrs)
                for rel in spatial_rels:
                    kg.add_spatial_relation(node.obj_id, rel)
            else:
                fb_reason = scene_reasoning
                extraction = TripleExtractor.extract_all(
                    reasoning=fb_reason,
                    category=q_cat,
                    queried_objects=[],
                    timestep=0,
                )
                attrs, spatial_rels = _attrs_and_spatial_from_extraction(
                    extraction,
                    fb_reason,
                    certainty=Certainty.LOW,
                )
                node = kg.add_object_merged(q_cat, bbox=None, timestep=0, image_id=sid)
                if attrs:
                    kg.update_attributes(node.obj_id, attrs)
                for rel in spatial_rels:
                    kg.add_spatial_relation(node.obj_id, rel)

        n_det += 1
        if n_det % 50 == 0:
            print(
                f"[build_global_kg] OVD passes: {n_det}, graph nodes: {kg.num_objects}",
                flush=True,
            )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_global_kg_bundle(kg, sample_id_to_hash, str(out))
    print(
        f"[build_global_kg] Wrote bundle {out} ({kg.num_objects} nodes, {n_det} OVD passes, "
        f"{len(sample_id_to_hash)} sample_id→hash entries)",
        flush=True,
    )
    return kg


def main() -> None:
    p = argparse.ArgumentParser(description="Build global IDKVQA KG (1 OVD VLM call per unique image)")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--split", type=str, default="val")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max unique images to process (each = one OVD call)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="results/global_kg.json")
    args = p.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    build_global_kg(
        cfg,
        split=args.split,
        limit=args.limit,
        seed=args.seed,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
