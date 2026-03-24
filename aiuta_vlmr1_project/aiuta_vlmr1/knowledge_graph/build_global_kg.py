"""
build_global_kg — Offline **global** knowledge graph for IDKVQA.

- **Dedup:** one detection VLM call per unique image (SHA-256 of PNG bytes).
- **Storage:** one ``ObjectNode`` per ``sample_id`` sharing that image; ``image_id`` is the
  IDKVQA ``sample_id`` string (same key as ``load_idkvqa`` / eval lookup).
- **Bundle JSON:** ``{format, graph, sample_id_to_image_hash}`` via
  :func:`aiuta_vlmr1.evaluation.global_kg_io.save_global_kg_bundle`.

**Oracle / leakage note:** building on the same split you evaluate is an upper-bound-style
analysis; document accordingly in papers.
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..config import Config
from ..evaluation.global_kg_io import save_global_kg_bundle
from ..evaluation.idkvqa_eval import (
    DETECTION_SYSTEM,
    _generate_chat,
    _idkvqa_eval_options,
    stable_idkvqa_image_id,
)
from ..evaluation.idkvqa_kg import enrich_kg_from_reasoning, subject_category_from_idkvqa_question
from ..evaluation.vlm_inference_utils import extract_answer_and_reasoning
from ..knowledge_graph.schema import Attribute, AttributeSource, Certainty
from ..utils.model_loader import ModelLoader
from .scene_graph import SceneKnowledgeGraph
from .triple_extractor import TripleExtractor

_DET_PROMPT = (
    "Carefully inspect this indoor scene. Describe all visible objects, their colors, "
    "materials, sizes, and spatial relationships."
)


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
        Maximum number of **unique images** to run detection on (after shuffling row order).
    """
    opts = _idkvqa_eval_options(config)
    det_tokens = int(opts["detection_max_new_tokens"])

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

    loader = ModelLoader.get_instance(config.model)
    kg = SceneKnowledgeGraph()
    n_det = 0

    for img_hash in ordered_hashes:
        if limit is not None and n_det >= limit:
            break
        members = groups[img_hash]
        row0 = members[0][1]
        pil_image = row0["image"]

        det_raw, _dt, _, _, _ = _generate_chat(
            loader,
            pil_image,
            DETECTION_SYSTEM,
            _DET_PROMPT,
            max_new_tokens=det_tokens,
            output_scores=False,
        )
        det_ans, det_think = extract_answer_and_reasoning(det_raw)
        reasoning = det_think or det_ans

        cat_seed_q = str(row0.get("question", ""))
        triple_category = subject_category_from_idkvqa_question(cat_seed_q)

        extraction = TripleExtractor.extract_all(
            reasoning=reasoning,
            category=triple_category,
            queried_objects=[],
            timestep=0,
        )
        kg_broad: dict[str, str] = {}
        for a in extraction.attributes:
            kg_broad[a.name] = a.value

        for at in ("color", "material"):
            enrich_kg_from_reasoning(kg_broad, reasoning, at)

        attrs_to_apply = [
            Attribute(
                name=k,
                value=str(v),
                certainty=Certainty.MEDIUM,
                source=AttributeSource.VLM_REASONING,
                timestep=0,
            )
            for k, v in kg_broad.items()
        ]

        for sid, row in members:
            qcat = subject_category_from_idkvqa_question(str(row.get("question", "")))
            node = kg.add_object_merged(qcat, bbox=None, timestep=0, image_id=sid)
            if attrs_to_apply:
                kg.update_attributes(node.obj_id, list(attrs_to_apply))
            for rel in extraction.spatial_relations:
                kg.add_spatial_relation(node.obj_id, rel)

        n_det += 1
        if n_det % 50 == 0:
            print(
                f"[build_global_kg] detection passes: {n_det}, graph nodes: {kg.num_objects}",
                flush=True,
            )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_global_kg_bundle(kg, sample_id_to_hash, str(out))
    print(
        f"[build_global_kg] Wrote bundle {out} ({kg.num_objects} nodes, {n_det} detection passes, "
        f"{len(sample_id_to_hash)} sample_id→hash entries)",
        flush=True,
    )
    return kg


def main() -> None:
    p = argparse.ArgumentParser(description="Build global IDKVQA KG (1 detection VLM call per unique image)")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--split", type=str, default="val")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max unique images to process (each = one detection call)",
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
