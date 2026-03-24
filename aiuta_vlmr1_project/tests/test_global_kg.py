"""Global KG serialization, lookup, and IDKVQA mode registration."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from aiuta_vlmr1.config import Config
from aiuta_vlmr1.detector.base import Detection, DetectionResult
from aiuta_vlmr1.evaluation.global_kg_io import load_global_kg_for_eval, save_global_kg_bundle
from aiuta_vlmr1.evaluation.idkvqa_eval import IDKVQA_MODES
from aiuta_vlmr1.evaluation.idkvqa_kg import subject_category_from_idkvqa_question
from aiuta_vlmr1.knowledge_graph.build_global_kg import (
    _pick_detection_for_question,
    _reasoning_slice_for_label,
    build_global_kg,
)
from aiuta_vlmr1.knowledge_graph.scene_graph import SceneKnowledgeGraph
from aiuta_vlmr1.knowledge_graph.schema import Attribute, Certainty


def test_scene_graph_serialization_roundtrip():
    kg = SceneKnowledgeGraph()
    node = kg.add_object("chair", bbox=[0.0, 0.0, 100.0, 100.0], timestep=0, image_id="img_001")
    kg.update_attributes(
        node.obj_id,
        [Attribute(name="color", value="red", certainty=Certainty.HIGH)],
    )

    data = kg.to_dict()
    kg2 = SceneKnowledgeGraph.from_dict(data)

    assert kg2.num_objects == 1
    node2 = kg2.get_objects_by_category("chair")[0]
    assert node2.get_attribute_value("color") == "red"
    assert node2.image_id == "img_001"


def test_get_attributes_for_image():
    kg = SceneKnowledgeGraph()
    n1 = kg.add_object("chair", image_id="img_001")
    kg.update_attributes(n1.obj_id, [Attribute(name="color", value="red", certainty=Certainty.HIGH)])
    n2 = kg.add_object("table", image_id="img_002")
    kg.update_attributes(n2.obj_id, [Attribute(name="color", value="brown", certainty=Certainty.HIGH)])

    attrs = kg.get_attributes_for_image("img_001")
    assert attrs == {"color": "red"}

    attrs2 = kg.get_attributes_for_image("img_002")
    assert attrs2 == {"color": "brown"}


def test_global_kg_in_modes():
    assert "global_kg" in IDKVQA_MODES
    assert "global_kg_entropy" in IDKVQA_MODES


def test_save_load_json(tmp_path: Path):
    kg = SceneKnowledgeGraph()
    kg.add_object("lamp", image_id="img_003")
    path = tmp_path / "test_kg.json"
    kg.save_json(path)
    kg2 = SceneKnowledgeGraph.load_json(path)
    assert kg2.num_objects == 1


def test_subject_category_from_question():
    assert subject_category_from_idkvqa_question("Is the chair red?") == "chair"
    assert subject_category_from_idkvqa_question("Is the red chair wooden?") == "chair"
    assert subject_category_from_idkvqa_question("Is there a lamp on the table?") == "lamp"


def test_global_kg_bundle_roundtrip(tmp_path: Path):
    kg = SceneKnowledgeGraph()
    n = kg.add_object("chair", image_id="42")
    kg.update_attributes(
        n.obj_id,
        [Attribute(name="color", value="red", certainty=Certainty.HIGH)],
    )
    mapping = {"42": "deadbeef", "43": "deadbeef"}
    path = tmp_path / "bundle.json"
    save_global_kg_bundle(kg, mapping, path)
    kg2, m2 = load_global_kg_for_eval(path)
    assert m2 == mapping
    assert kg2.num_objects == 1
    assert kg2.get_attributes_for_image("42") == {"color": "red"}


def test_load_global_kg_legacy_pure_graph(tmp_path: Path):
    kg = SceneKnowledgeGraph()
    kg.add_object("table", image_id="7")
    path = tmp_path / "legacy.json"
    kg.save_json(path)
    kg2, m = load_global_kg_for_eval(path)
    assert m == {}
    assert kg2.num_objects == 1


def test_reasoning_slice_prefers_label_sentences():
    r = "A lamp is tall. The chair is red. The table is blue."
    s = _reasoning_slice_for_label(r, "chair")
    assert "chair" in s.lower() and "red" in s.lower()
    assert "table" not in s.lower() or "lamp" not in s  # chair-only slice


def test_pick_detection_matches_label():
    d1 = Detection(bbox=[0, 0, 1, 1], label="chair")
    d2 = Detection(bbox=[1, 1, 2, 2], label="table")
    assert _pick_detection_for_question([d1, d2], "chair") is d1
    assert _pick_detection_for_question([d1, d2], "table") is d2


def test_build_global_kg_per_object_distinct_attrs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Same image, two questions (chair vs table): KG attrs differ per sample_id when OVD returns two dets."""
    img = Image.new("RGB", (64, 64), color=(100, 100, 100))
    rows = [
        {"id": "s1", "image": img, "question": "Is the chair white?"},
        {"id": "s2", "image": img, "question": "Is the table red?"},
    ]

    monkeypatch.setattr(
        "aiuta_vlmr1.knowledge_graph.build_global_kg._load_hf_rows",
        lambda _split, _seed: rows,
    )
    monkeypatch.setattr(
        "aiuta_vlmr1.knowledge_graph.build_global_kg.ModelLoader",
        MagicMock(get_instance=lambda *_a, **_kw: object()),
    )

    d_chair = Detection(bbox=[0.0, 0.0, 10.0, 10.0], label="chair", reasoning="")
    d_table = Detection(bbox=[20.0, 20.0, 40.0, 40.0], label="table", reasoning="")

    class _FakeDet:
        def __init__(self, _cfg):
            pass

        def detect_from_observation(self, _obs, _categories, kg_context=None):
            return DetectionResult(
                detections=[d_chair, d_table],
                raw_output="",
                reasoning_text=(
                    "The chair in the scene is white. The wooden table is painted red."
                ),
                json_valid=True,
                latency_sec=0.0,
            )

    monkeypatch.setattr(
        "aiuta_vlmr1.knowledge_graph.build_global_kg.VLMr1Detector",
        _FakeDet,
    )

    out = tmp_path / "bundle.json"
    kg = build_global_kg(Config(), split="val", limit=1, seed=0, output_path=str(out))
    assert kg.num_objects == 2
    a1 = kg.get_attributes_for_image("s1")
    a2 = kg.get_attributes_for_image("s2")
    assert a1 != a2, (a1, a2)
    assert "color" in a1 and "color" in a2
