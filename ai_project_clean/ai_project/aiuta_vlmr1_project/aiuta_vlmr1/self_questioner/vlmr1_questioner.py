"""vlmr1_questioner.py — Self-Questioner using VLM-R1 reasoning. 0 extra calls."""
from __future__ import annotations
from ..detector.base import Detection
from ..knowledge_graph.schema import TargetFacts
from ..knowledge_graph.scene_graph import SceneKnowledgeGraph
from ..knowledge_graph.triple_extractor import TripleExtractor
from .base import AbstractSelfQuestioner, RefinedDescription

class VLMr1SelfQuestioner(AbstractSelfQuestioner):
    def process(self, detection: Detection, target_facts: TargetFacts,
                kg: SceneKnowledgeGraph, timestep: int = 0) -> RefinedDescription:
        if not detection.reasoning:
            return RefinedDescription(object_node=None, text_description="", is_valid=False)
        extraction = TripleExtractor.extract_all(
            reasoning=detection.reasoning, category=detection.label,
            queried_objects=[], timestep=timestep,
        )
        node = kg.add_object_merged(category=detection.label, bbox=detection.bbox, timestep=timestep)
        if extraction.attributes:
            kg.update_attributes(node.obj_id, extraction.attributes)
        for rel in extraction.spatial_relations:
            kg.add_spatial_relation(node.obj_id, rel)
        return RefinedDescription(
            object_node=node, text_description=node.to_natural_language(), is_valid=True,
        )
