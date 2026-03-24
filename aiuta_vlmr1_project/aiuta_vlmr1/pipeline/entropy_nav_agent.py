"""
Entropy-first navigation with dual KG (global + episodic) entropy modulation.

Separate from :mod:`entropy_coin_agent` (backward compatible). Uses
:func:`aiuta_vlmr1.evaluation.idkvqa_kg.compute_dual_kg_entropy_modulation`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..evaluation.answer_normalization import LABEL_IDK, normalize_yes_no_idk
from ..evaluation.idkvqa_kg import (
    compute_dual_kg_entropy_modulation,
    enrich_kg_from_reasoning,
    parse_question_attribute,
)
from ..evaluation.vlm_inference_utils import extract_answer_and_reasoning
from ..knowledge_graph.scene_graph import SceneKnowledgeGraph
from ..knowledge_graph.schema import Attribute, AttributeSource, Certainty
from ..knowledge_graph.triple_extractor import TripleExtractor
from .entropy_coin_agent import vlm_vqa_with_entropy

VQA_SYSTEM = (
    "You are a helpful assistant that answers visual questions about images. "
    "You must reason carefully about what you can see in the image. "
    "The reasoning process must be enclosed within <think></think> tags, "
    "and the final answer must be enclosed within <answer> </answer> tags."
)


@dataclass
class NavStepLog:
    step: int
    answer: str
    raw_entropy: float | None
    adjusted_entropy: float | None
    kg_signal: dict[str, str]
    action: str
    episode_kg_nodes: int
    attributes_extracted: dict[str, str]


@dataclass
class NavEpisodeResult:
    episode_id: str
    target_category: str
    success: bool
    final_answer: str | None
    steps: int
    path_length: float
    shortest_path_length: float
    num_questions: int
    entropy_trajectory: list[float]
    step_logs: list[NavStepLog]
    stagnation_exit: bool


def _global_attrs_for_category(kg: SceneKnowledgeGraph, target_category: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for node in kg.get_objects_by_category(target_category):
        for name, attr in node.attributes.items():
            if name not in out:
                out[name] = attr.value
        for rel in node.spatial_relations:
            if rel.relation not in out:
                out[rel.relation] = rel.reference
    return out


def _episode_attrs_flat(episode_kg: SceneKnowledgeGraph) -> dict[str, str]:
    out: dict[str, str] = {}
    for node in episode_kg.all_objects():
        for name, attr in node.attributes.items():
            if name not in out:
                out[name] = attr.value
        for rel in node.spatial_relations:
            if rel.relation not in out:
                out[rel.relation] = rel.reference
    return out


def _extract_and_accumulate_kg(
    reasoning: str,
    target_category: str,
    episode_kg: SceneKnowledgeGraph,
    image_id: str,
) -> dict[str, str]:
    extraction = TripleExtractor.extract_all(
        reasoning=reasoning,
        category=target_category,
        queried_objects=[],
        timestep=0,
    )
    attrs_dict: dict[str, str] = {}
    for a in extraction.attributes:
        attrs_dict[a.name] = a.value
    for at in ("color", "material"):
        enrich_kg_from_reasoning(attrs_dict, reasoning, at)

    if attrs_dict:
        node = episode_kg.add_object_merged(
            target_category, bbox=None, timestep=0, image_id=image_id,
        )
        attrs_to_apply = [
            Attribute(
                name=k,
                value=str(v),
                certainty=Certainty.MEDIUM,
                source=AttributeSource.VLM_REASONING,
                timestep=0,
            )
            for k, v in attrs_dict.items()
        ]
        episode_kg.update_attributes(node.obj_id, attrs_to_apply)
        for rel in extraction.spatial_relations:
            episode_kg.add_spatial_relation(node.obj_id, rel)

    return attrs_dict


def run_nav_episode(
    env: Any,
    model: Any,
    processor: Any,
    question: str,
    target_category: str,
    global_kg: SceneKnowledgeGraph | None = None,
    *,
    tau: float = 0.15,
    max_steps: int = 20,
    stagnation_patience: int = 3,
    stagnation_epsilon: float = 0.01,
    boost_factor: float = 0.5,
    penalty_factor: float = 2.0,
    entropy_cap: float = 0.30,
    max_new_tokens: int = 256,
) -> NavEpisodeResult:
    """
    One navigation episode: VLM + episodic KG accumulation + dual KG entropy modulation.
    """
    episode_kg = SceneKnowledgeGraph()
    entropy_trajectory: list[float] = []
    step_logs: list[NavStepLog] = []
    best_entropy = float("inf")
    stagnation_count = 0
    attr_type, attr_value = parse_question_attribute(question)

    global_attrs: dict[str, str] = {}
    if global_kg is not None and target_category:
        global_attrs = _global_attrs_for_category(global_kg, target_category)

    for step in range(max_steps):
        observation = env.get_observation()
        answer_text, raw_entropy, _logits = vlm_vqa_with_entropy(
            image=observation,
            question=question,
            model=model,
            processor=processor,
            max_new_tokens=max_new_tokens,
            system_prompt=VQA_SYSTEM,
        )
        raw_answer, reasoning = extract_answer_and_reasoning(answer_text)
        normalized = normalize_yes_no_idk(raw_answer)

        extracted_attrs = _extract_and_accumulate_kg(
            reasoning or "",
            target_category,
            episode_kg,
            image_id=f"step_{step}",
        )
        episode_attrs = _episode_attrs_flat(episode_kg)

        _, adjusted_entropy, signals = compute_dual_kg_entropy_modulation(
            normalized,
            global_attrs,
            episode_attrs,
            attr_type,
            attr_value,
            raw_entropy,
            boost_factor=boost_factor,
            penalty_factor=penalty_factor,
            cap=entropy_cap,
        )

        eff_entropy = adjusted_entropy if adjusted_entropy is not None else (raw_entropy if raw_entropy is not None else 1.0)
        entropy_trajectory.append(float(eff_entropy))

        if eff_entropy < float(tau) and normalized != LABEL_IDK:
            step_logs.append(
                NavStepLog(
                    step=step,
                    answer=normalized,
                    raw_entropy=raw_entropy,
                    adjusted_entropy=adjusted_entropy,
                    kg_signal=signals,
                    action="commit",
                    episode_kg_nodes=episode_kg.num_objects,
                    attributes_extracted=extracted_attrs,
                )
            )
            return NavEpisodeResult(
                episode_id=env.episode_id,
                target_category=env.target_category,
                success=env.evaluate_commit(normalized),
                final_answer=normalized,
                steps=step + 1,
                path_length=env.path_length,
                shortest_path_length=float(env.shortest_path_length),
                num_questions=0,
                entropy_trajectory=entropy_trajectory,
                step_logs=step_logs,
                stagnation_exit=False,
            )

        if eff_entropy < best_entropy - stagnation_epsilon:
            best_entropy = eff_entropy
            stagnation_count = 0
        else:
            stagnation_count += 1

        if stagnation_count >= stagnation_patience:
            step_logs.append(
                NavStepLog(
                    step=step,
                    answer=normalized,
                    raw_entropy=raw_entropy,
                    adjusted_entropy=adjusted_entropy,
                    kg_signal=signals,
                    action="abstain",
                    episode_kg_nodes=episode_kg.num_objects,
                    attributes_extracted=extracted_attrs,
                )
            )
            return NavEpisodeResult(
                episode_id=env.episode_id,
                target_category=env.target_category,
                success=False,
                final_answer=LABEL_IDK,
                steps=step + 1,
                path_length=env.path_length,
                shortest_path_length=float(env.shortest_path_length),
                num_questions=0,
                entropy_trajectory=entropy_trajectory,
                step_logs=step_logs,
                stagnation_exit=True,
            )

        step_logs.append(
            NavStepLog(
                step=step,
                answer=normalized,
                raw_entropy=raw_entropy,
                adjusted_entropy=adjusted_entropy,
                kg_signal=signals,
                action="explore",
                episode_kg_nodes=episode_kg.num_objects,
                attributes_extracted=extracted_attrs,
            )
        )

        if getattr(env, "is_exhausted", False):
            return NavEpisodeResult(
                episode_id=env.episode_id,
                target_category=env.target_category,
                success=env.evaluate_commit(normalized),
                final_answer=normalized,
                steps=step + 1,
                path_length=env.path_length,
                shortest_path_length=float(env.shortest_path_length),
                num_questions=0,
                entropy_trajectory=entropy_trajectory,
                step_logs=step_logs,
                stagnation_exit=False,
            )

        env.step("explore")

    return NavEpisodeResult(
        episode_id=env.episode_id,
        target_category=env.target_category,
        success=False,
        final_answer=LABEL_IDK,
        steps=max_steps,
        path_length=env.path_length,
        shortest_path_length=float(env.shortest_path_length),
        num_questions=0,
        entropy_trajectory=entropy_trajectory,
        step_logs=step_logs,
        stagnation_exit=False,
    )
