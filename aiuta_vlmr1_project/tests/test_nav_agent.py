"""Tests for entropy_nav_agent (mocked VLM)."""
from __future__ import annotations

from PIL import Image

from aiuta_vlmr1.pipeline.entropy_nav_agent import run_nav_episode


class _NavEnvStub:
    """Minimal env matching CoINBenchEnv interface for unit tests."""

    episode_id = "ep1"
    target_category = "chair"

    def __init__(self, images: list):
        self._imgs = images
        self._nav = 0

    def get_observation(self):
        return self._imgs[min(self._nav, len(self._imgs) - 1)]

    def step(self, action: str) -> None:
        self._nav += 1

    def evaluate_commit(self, a: str) -> bool:
        return "yes" in a.lower()

    @property
    def path_length(self) -> float:
        return float(self._nav + 1)

    @property
    def shortest_path_length(self) -> float:
        return 1.0

    @property
    def is_exhausted(self) -> bool:
        return self._nav >= len(self._imgs) - 1


def test_nav_episode_commits_when_confident(monkeypatch):
    def _stub(*_a, **_k):
        return "<think>It is red.</think><answer>Yes</answer>", 0.02, None

    monkeypatch.setattr(
        "aiuta_vlmr1.pipeline.entropy_nav_agent.vlm_vqa_with_entropy",
        _stub,
    )
    img = Image.new("RGB", (8, 8), color="red")
    env = _NavEnvStub([img])
    result = run_nav_episode(
        env,
        model=None,
        processor=None,
        question="Is this the red chair?",
        target_category="chair",
        global_kg=None,
        tau=0.15,
        max_steps=5,
    )
    assert result.success is True
    assert result.steps == 1
    assert result.num_questions == 0
    assert result.step_logs[-1].action == "commit"


def test_nav_episode_explores_then_commits(monkeypatch):
    call_count = [0]

    def _stub(*_a, **_k):
        call_count[0] += 1
        if call_count[0] <= 2:
            return "<think>unclear</think><answer>Yes</answer>", 0.25, None
        return "<think>Yes red chair.</think><answer>Yes</answer>", 0.03, None

    monkeypatch.setattr(
        "aiuta_vlmr1.pipeline.entropy_nav_agent.vlm_vqa_with_entropy",
        _stub,
    )
    imgs = [Image.new("RGB", (4, 4), color=c) for c in ("red", "green", "blue")]
    env = _NavEnvStub(imgs)
    result = run_nav_episode(
        env,
        model=None,
        processor=None,
        question="Is this the red chair?",
        target_category="chair",
        global_kg=None,
        tau=0.15,
        max_steps=5,
    )
    assert result.steps == 3
    assert result.step_logs[0].action == "explore"
    assert result.step_logs[2].action == "commit"


def test_nav_episode_stagnation_exit(monkeypatch):
    def _stub(*_a, **_k):
        return "<think>not sure</think><answer>Yes</answer>", 0.20, None

    monkeypatch.setattr(
        "aiuta_vlmr1.pipeline.entropy_nav_agent.vlm_vqa_with_entropy",
        _stub,
    )
    imgs = [Image.new("RGB", (4, 4), color="gray") for _ in range(5)]
    env = _NavEnvStub(imgs)
    result = run_nav_episode(
        env,
        model=None,
        processor=None,
        question="Is this?",
        target_category="chair",
        global_kg=None,
        tau=0.15,
        max_steps=10,
        stagnation_patience=2,
        stagnation_epsilon=0.01,
    )
    assert result.stagnation_exit is True
    assert result.final_answer == "I don't know"
    assert result.success is False
    assert result.steps == 3
