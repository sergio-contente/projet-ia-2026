"""
question_generator.py — Generate discriminative questions for the user.
Replaces AIUTA LLM-based question generation with KG analysis.
"""
from __future__ import annotations
from .schema import ObjectNode, TargetFacts
from .scene_graph import SceneKnowledgeGraph
from .graph_matcher import GraphMatcher

QUESTION_TEMPLATES = {
    "color": "What color is the {category}?",
    "material": "What material is the {category} made of?",
    "size": "Is the {category} large or small?",
    "has_glass_door": "Does the {category} have a glass door?",
    "has_handle": "Does the {category} have a handle?",
    "has_drawer": "Does the {category} have drawers?",
    "is_open": "Is the {category} open or closed?",
    "location": "In which room is the {category} located?",
    "near": "What is the {category} near or next to?",
}

GENERIC_QUESTIONS = [
    "Can you describe any distinctive features of the {category}?",
    "What does the {category} look like?",
]


class QuestionGenerator:
    @staticmethod
    def generate(obj: ObjectNode, kg: SceneKnowledgeGraph) -> str:
        all_instances = kg.get_objects_by_category(obj.category)
        target = kg.target_facts
        all_attr_names = set()
        for inst in all_instances:
            all_attr_names.update(inst.attributes.keys())
        known = set(target.known_attributes.keys()) | set(target.negative_attributes.keys())
        unknown = [a for a in all_attr_names if a not in known]
        for common in ["color", "material", "size", "location", "near"]:
            if common not in known and common not in unknown:
                unknown.append(common)
        if not unknown:
            return GENERIC_QUESTIONS[0].format(category=obj.category)
        ranked = [(a, GraphMatcher.compute_discriminative_power(a, all_instances)) for a in unknown]
        ranked.sort(key=lambda x: -x[1])
        asked = set(target.asked_questions)
        for attr_name, _score in ranked:
            template = QUESTION_TEMPLATES.get(attr_name, f"What is the {attr_name} of the {{category}}?")
            candidate = template.format(category=obj.category)
            if candidate not in asked:
                return candidate
        for g in GENERIC_QUESTIONS:
            candidate = g.format(category=obj.category)
            if candidate not in asked:
                return candidate
        best = ranked[0][0]
        template = QUESTION_TEMPLATES.get(best, f"What is the {best} of the {{category}}?")
        return template.format(category=obj.category)
