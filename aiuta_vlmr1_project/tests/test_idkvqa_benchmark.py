"""Tests for primary IDKVQA benchmark (no GPU)."""
from __future__ import annotations

from aiuta_vlmr1.evaluation.answer_normalization import LABEL_IDK, LABEL_NO, LABEL_YES, normalize_yes_no_idk
from aiuta_vlmr1.evaluation.idkvqa_eval import IDKVQA_MODES, finalize_for_mode
from aiuta_vlmr1.evaluation.idkvqa_types import QAExampleResult, aggregate_idkvqa_metrics
from aiuta_vlmr1.evaluation.mode_transition_analysis import compare_mode_transitions
from aiuta_vlmr1.evaluation.normalization_audit import export_normalization_audit, select_audit_indices
from aiuta_vlmr1.evaluation.paper_artifacts import export_mode_tables, main_benchmark_row, reliability_row
from aiuta_vlmr1.evaluation.question_type_analysis import metrics_by_mode_and_question_type
from aiuta_vlmr1.evaluation.threshold_sweep import sweep_entropy_threshold


def test_normalize_variants():
    assert normalize_yes_no_idk("yeah, visible") == LABEL_YES
    assert normalize_yes_no_idk("nope, absent") == LABEL_NO
    assert normalize_yes_no_idk("unclear") == LABEL_IDK
    assert normalize_yes_no_idk("cannot tell") == LABEL_IDK


def test_raw_two_pass_in_modes():
    assert "raw_two_pass" in IDKVQA_MODES


def test_two_pass_kg_relaxed_in_modes():
    assert "two_pass_kg_relaxed" in IDKVQA_MODES


def test_finalize_modes():
    r = finalize_for_mode(
        "raw",
        raw_normalized=LABEL_YES,
        uncertainty_score=0.9,
        threshold=0.5,
        rule="entropy_above_tau_to_idk",
        kg_hybrid=None,
    )
    assert r[0] == LABEL_YES and r[1] is False

    r2 = finalize_for_mode(
        "threshold",
        raw_normalized=LABEL_YES,
        uncertainty_score=0.9,
        threshold=0.5,
        rule="entropy_above_tau_to_idk",
        kg_hybrid=None,
    )
    assert r2[0] == LABEL_IDK and r2[3] is True

    r3 = finalize_for_mode(
        "kg",
        raw_normalized=LABEL_NO,
        uncertainty_score=0.1,
        threshold=0.5,
        rule="entropy_above_tau_to_idk",
        kg_hybrid=LABEL_YES,
    )
    assert r3[0] == LABEL_YES and r3[1] is True

    r4 = finalize_for_mode(
        "kg_threshold",
        raw_normalized=LABEL_NO,
        uncertainty_score=0.9,
        threshold=0.5,
        rule="entropy_above_tau_to_idk",
        kg_hybrid=LABEL_YES,
    )
    assert r4[0] == LABEL_IDK

    r5 = finalize_for_mode(
        "raw_two_pass",
        raw_normalized=LABEL_NO,
        uncertainty_score=0.2,
        threshold=0.5,
        rule="entropy_above_tau_to_idk",
        kg_hybrid=None,
    )
    assert r5[0] == LABEL_NO and r5[1] is False

    r6 = finalize_for_mode(
        "two_pass_kg",
        raw_normalized=LABEL_NO,
        uncertainty_score=0.5,
        threshold=0.5,
        rule="entropy_above_tau_to_idk",
        kg_hybrid=LABEL_YES,
    )
    assert r6[0] == LABEL_YES and r6[1] is True and r6[2] is False

    r7 = finalize_for_mode(
        "two_pass_kg_relaxed",
        raw_normalized=LABEL_NO,
        uncertainty_score=0.5,
        threshold=0.5,
        rule="entropy_above_tau_to_idk",
        kg_hybrid=LABEL_YES,
    )
    assert r7[0] == LABEL_YES and r7[1] is True and r7[2] is False


def test_overclaim_underclaim_metrics():
    results = [
        QAExampleResult(
            sample_id="1",
            question="q",
            ground_truth=LABEL_IDK,
            raw_prediction="x",
            final_prediction=LABEL_YES,
            confidence_score=None,
            entropy_score=None,
            used_kg=False,
            used_threshold=True,
            used_abstention=False,
            latency_sec=0.1,
            metadata={},
        ),
        QAExampleResult(
            sample_id="2",
            question="q",
            ground_truth=LABEL_YES,
            raw_prediction="x",
            final_prediction=LABEL_IDK,
            confidence_score=None,
            entropy_score=None,
            used_kg=False,
            used_threshold=True,
            used_abstention=True,
            latency_sec=0.1,
            metadata={},
        ),
    ]
    m = aggregate_idkvqa_metrics(results)
    cal = m["calibration_abstention"]
    assert cal["overclaim_rate"] > 0
    assert cal["underclaim_rate"] > 0
    ca = m["cost_accounting"]
    assert "avg_num_model_calls" in ca


def _synthetic_result(
    sid: str,
    gt: str,
    final: str,
    raw: str = "draft",
    mode: str = "raw",
    entropy: float | None = 0.2,
) -> QAExampleResult:
    fn = normalize_yes_no_idk(final)
    return QAExampleResult(
        sample_id=sid,
        question="Is there a table?",
        ground_truth=gt,
        raw_prediction=raw,
        final_prediction=final,
        confidence_score=0.9,
        entropy_score=entropy,
        used_kg=False,
        used_threshold=False,
        used_abstention=False,
        latency_sec=1.0,
        metadata={"raw_prediction_normalized": normalize_yes_no_idk(raw)},
        question_type="existence",
        mode=mode,
        raw_prediction_label=normalize_yes_no_idk(raw),
        correct=(fn == gt),
        num_model_calls=1,
        num_detector_calls=0,
        num_questioner_calls=0,
        num_trigger_calls=0,
        num_questions_asked=1,
        num_kg_nodes=None,
        total_latency_sec=1.0,
        detector_latency_sec=None,
        decision_latency_sec=0.0,
        uncertainty_score=entropy,
        threshold=None,
        abstained=(fn == LABEL_IDK),
    )


def test_compare_mode_transitions():
    base = [
        _synthetic_result("a", LABEL_YES, LABEL_YES),
        _synthetic_result("b", LABEL_YES, LABEL_NO),
        _synthetic_result("c", LABEL_IDK, LABEL_YES),
    ]
    imp = [
        _synthetic_result("a", LABEL_YES, LABEL_YES),
        _synthetic_result("b", LABEL_YES, LABEL_YES),
        _synthetic_result("c", LABEL_IDK, LABEL_IDK),
    ]
    t = compare_mode_transitions(base, imp)
    assert t["num_aligned_samples"] == 3
    assert t["baseline_correct_to_improved_correct"] == 1
    # ``b``: No→Yes; ``c``: overclaim Yes→IDK when GT is IDK.
    assert t["baseline_wrong_to_improved_correct"] == 2


def test_question_type_aggregation():
    r1 = _synthetic_result("1", LABEL_YES, LABEL_YES)
    r2 = _synthetic_result("2", LABEL_NO, LABEL_NO)
    m = metrics_by_mode_and_question_type({"raw": [r1, r2]})
    assert m["question_type_source"] == "heuristic_coarse_taxonomy"
    assert "raw" in m["modes"]


def test_normalization_audit_exporter(tmp_path):
    results = [
        _synthetic_result("1", LABEL_YES, LABEL_YES, raw="yes"),
        _synthetic_result("2", LABEL_NO, LABEL_YES, raw="maybe"),
    ]
    p = tmp_path / "audit.json"
    export_normalization_audit(results, p, n=2, strategy="random", seed=0)
    assert p.is_file()
    idx = select_audit_indices(results, 1, "first_mismatches")
    assert idx == [1]


def test_paper_artifact_schemas(tmp_path):
    r = _synthetic_result("1", LABEL_YES, LABEL_YES)
    m = aggregate_idkvqa_metrics([r])
    assert "accuracy_pct" in main_benchmark_row("raw", m)
    assert "phi_c1_pct" in reliability_row("raw", m)
    export_mode_tables({"raw": [r]}, tmp_path)
    assert (tmp_path / "paper_main_benchmark.json").is_file()


def test_threshold_sweep_schema():
    raw_results = [
        _synthetic_result("1", LABEL_YES, LABEL_YES, raw="Yes", entropy=0.1),
        _synthetic_result("2", LABEL_NO, LABEL_NO, raw="No", entropy=0.8),
    ]
    rows = sweep_entropy_threshold(raw_results, [0.05, 0.5])
    assert len(rows) == 2
    assert "metrics" in rows[0]
    assert rows[0]["metrics"]["num_samples"] == 2
