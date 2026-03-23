"""scene_graph.py — Per-episode scene knowledge graph. Repository pattern."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterator

from ..evaluation.coin_metrics import compute_iou
from .schema import Attribute, ObjectNode, SpatialRelation, TargetFacts, Certainty


class SceneKnowledgeGraph:
    """
    Object deduplication (merge) heuristic:

    When adding a detection of the same ``category`` with bounding boxes whose IoU exceeds
    ``merge_iou_threshold`` and timesteps within ``merge_timestep_window`` of an existing
    node, we update ``timestep_last`` and return the existing node instead of creating a
    duplicate. This reduces redundant nodes from repeated detections of the same instance.
    """

    def __init__(self, merge_iou_threshold: float = 0.5, merge_timestep_window: int = 1):
        self._nodes: dict[str, ObjectNode] = {}
        self._category_index: dict[str, list[str]] = defaultdict(list)
        self._id_counter: dict[str, int] = defaultdict(int)
        self.target_facts = TargetFacts()
        self._merge_iou_threshold = merge_iou_threshold
        self._merge_timestep_window = merge_timestep_window

    def add_object(self, category: str, bbox=None, timestep: int = 0, image_id=None) -> ObjectNode:
        self._id_counter[category] += 1
        obj_id = f"{category}_{self._id_counter[category]:03d}"
        node = ObjectNode(
            obj_id=obj_id,
            category=category,
            bbox=bbox,
            image_id=image_id,
            timestep_first=timestep,
            timestep_last=timestep,
        )
        self._nodes[obj_id] = node
        self._category_index[category].append(obj_id)
        return node

    def add_object_merged(
        self,
        category: str,
        bbox: list[float] | None = None,
        timestep: int = 0,
        image_id=None,
    ) -> ObjectNode:
        """
        Add a new object or merge with an existing same-category node if bbox IoU is high
        and timestep is close (see class docstring).
        """
        if bbox is not None and len(bbox) == 4:
            for oid in self._category_index.get(category, []):
                node = self._nodes.get(oid)
                if node is None or node.bbox is None or len(node.bbox) != 4:
                    continue
                try:
                    iou = compute_iou(bbox, node.bbox)
                except (TypeError, ValueError):
                    continue
                if iou >= self._merge_iou_threshold:
                    if abs(timestep - node.timestep_last) <= self._merge_timestep_window:
                        node.timestep_last = timestep
                        if image_id:
                            node.image_id = image_id
                        return node
        return self.add_object(category, bbox=bbox, timestep=timestep, image_id=image_id)

    def update_attributes(self, obj_id: str, attributes: list[Attribute]) -> None:
        node = self._nodes.get(obj_id)
        if node is None:
            raise KeyError(f"Object {obj_id} not found")
        for attr in attributes:
            existing = node.attributes.get(attr.name)
            if existing is None or attr.certainty > existing.certainty:
                node.attributes[attr.name] = attr

    def add_spatial_relation(self, obj_id: str, relation: SpatialRelation) -> None:
        node = self._nodes.get(obj_id)
        if node is None:
            raise KeyError(f"Object {obj_id} not found")
        node.spatial_relations.append(relation)

    def get_object(self, obj_id: str) -> ObjectNode | None:
        return self._nodes.get(obj_id)

    def get_objects_by_category(self, category: str) -> list[ObjectNode]:
        ids = self._category_index.get(category, [])
        return [self._nodes[oid] for oid in ids if oid in self._nodes]

    def all_objects(self) -> Iterator[ObjectNode]:
        return iter(self._nodes.values())

    def update_target_facts(self, user_response: str, timestep: int = 0) -> None:
        from .target_fact_parser import parse_user_response_to_facts

        facts = parse_user_response_to_facts(user_response)
        src = f"user_t{timestep}"
        structured = [f for f in facts if f.provenance != "fallback"]
        if structured:
            for f in structured:
                if f.negative:
                    self.target_facts.add_negative(f.attribute, f.value, f"{f.provenance}|{src}")
                else:
                    self.target_facts.add_positive(f.attribute, f.value, f"{f.provenance}|{src}")
            return

        r = user_response.strip().lower()
        is_neg = any(r.startswith(p) for p in ("no", "not", "it is not", "it\'s not"))
        if is_neg:
            clean = r
            for prefix in ("no, ", "not ", "it is not ", "it\'s not "):
                if clean.startswith(prefix):
                    clean = clean[len(prefix):]
                    break
            self.target_facts.add_negative("user_stated", clean, src)
        else:
            clean = r
            for prefix in ("yes, ", "it is ", "it\'s ", "it has "):
                if clean.startswith(prefix):
                    clean = clean[len(prefix):]
                    break
            self.target_facts.add_positive("user_stated", clean, src)

    def get_kg_context_string(self, category: str) -> str:
        instances = self.get_objects_by_category(category)
        if not instances:
            return ""
        lines = [f"Previously observed {len(instances)} {category}(s):"]
        for node in instances:
            lines.append(f"  - {node.to_natural_language()}")
        if self.target_facts.num_facts > 0:
            lines.append(f"\nTarget: {self.target_facts.to_natural_language()}")
        return "\n".join(lines)

    def get_missing_attributes(self, obj_id: str) -> list[str]:
        node = self._nodes.get(obj_id)
        if node is None:
            return []
        return [a for a in self.target_facts.known_attributes if not node.has_attribute(a)]

    def reset(self):
        self._nodes.clear()
        self._category_index.clear()
        self._id_counter.clear()
        self.target_facts = TargetFacts()

    @property
    def num_objects(self) -> int:
        return len(self._nodes)
