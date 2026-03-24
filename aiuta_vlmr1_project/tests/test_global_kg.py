"""Global KG serialization, lookup, and IDKVQA mode registration."""
from __future__ import annotations

from pathlib import Path

from aiuta_vlmr1.evaluation.global_kg_io import load_global_kg_for_eval, save_global_kg_bundle
from aiuta_vlmr1.evaluation.idkvqa_eval import IDKVQA_MODES
from aiuta_vlmr1.evaluation.idkvqa_kg import subject_category_from_idkvqa_question
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
