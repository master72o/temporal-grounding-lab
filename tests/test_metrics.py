"""
Unit tests for Temporal Grounding metrics.
"""

import pytest
from temporalgrounding.metrics import (
    compute_tiou,
    compute_boundary_jitter,
    check_hallucination,
    compute_temporal_ordering_accuracy,
    compute_recall_auc,
    diagnose_boundary_bias,
    temporal_nms,
    evaluate_event_prediction,
)
from temporalgrounding.schema import GroundTruthEvent, PredictedInterval
from temporalgrounding.schema import ModelTier


def test_tiou_exact_match():
    assert compute_tiou(10.0, 20.0, 10.0, 20.0) == 1.0


def test_tiou_disjoint():
    assert compute_tiou(10.0, 20.0, 25.0, 35.0) == 0.0


def test_tiou_partial_overlap():
    # Intersection: [15, 20] -> 5.0
    # Union: [10, 25] -> 15.0
    # tIoU: 5 / 15 = 0.3333...
    assert pytest.approx(compute_tiou(10.0, 20.0, 15.0, 25.0), 0.001) == 1.0 / 3.0


def test_boundary_jitter():
    # start err: |12 - 10| = 2
    # end err: |18 - 20| = 2
    # jitter: (2 + 2) / 2 = 2.0
    assert compute_boundary_jitter(12.0, 18.0, 10.0, 20.0) == 2.0


def test_hallucination_zero_overlap():
    assert check_hallucination(0.0, 50.0, 60.0, 100.0) is True


def test_hallucination_outside_video():
    assert check_hallucination(0.5, 105.0, 110.0, 100.0) is True


def test_temporal_ordering_concordance():
    gt_events = [
        GroundTruthEvent(event_id="E1", label="First", start_sec=10.0, end_sec=15.0, description=""),
        GroundTruthEvent(event_id="E2", label="Second", start_sec=20.0, end_sec=25.0, description=""),
        GroundTruthEvent(event_id="E3", label="Third", start_sec=30.0, end_sec=35.0, description=""),
    ]
    # Correct order
    preds_correct = [
        PredictedInterval(event_id="E1", label="First", start_sec=11.0, end_sec=16.0),
        PredictedInterval(event_id="E2", label="Second", start_sec=21.0, end_sec=26.0),
        PredictedInterval(event_id="E3", label="Third", start_sec=31.0, end_sec=36.0),
    ]
    assert compute_temporal_ordering_accuracy(preds_correct, gt_events) == 1.0

    # Inverted order
    preds_inverted = [
        PredictedInterval(event_id="E1", label="First", start_sec=31.0, end_sec=36.0),
        PredictedInterval(event_id="E2", label="Second", start_sec=21.0, end_sec=26.0),
        PredictedInterval(event_id="E3", label="Third", start_sec=11.0, end_sec=16.0),
    ]
    assert compute_temporal_ordering_accuracy(preds_inverted, gt_events) == 0.0


def test_recall_auc_and_boundary_bias_diagnostics():
    gt = GroundTruthEvent(event_id="E1", label="Action", start_sec=10.0, end_sec=20.0, description="")
    pred_tight = PredictedInterval(event_id="E1", label="Action", start_sec=10.2, end_sec=19.8)
    eval_res = evaluate_event_prediction(pred_tight, gt, 60.0, ModelTier.TEMPORAL_SENTINEL)

    auc_res = compute_recall_auc([eval_res])
    assert auc_res["auc"] == 1.0

    assert diagnose_boundary_bias(10.2, 19.8, 10.0, 20.0) == "tight_localization"
    assert diagnose_boundary_bias(5.0, 25.0, 10.0, 20.0) == "over_extended"
    assert diagnose_boundary_bias(13.0, 17.0, 10.0, 20.0) == "under_extended"
    assert diagnose_boundary_bias(40.0, 50.0, 10.0, 20.0) == "hallucinated_disjoint"


def test_temporal_nms_deduplication():
    proposals = [
        PredictedInterval(event_id="P1", label="Goal", start_sec=10.0, end_sec=20.0, confidence=0.95),
        PredictedInterval(event_id="P2", label="Goal", start_sec=11.0, end_sec=21.0, confidence=0.70),  # High overlap
        PredictedInterval(event_id="P3", label="Goal", start_sec=40.0, end_sec=50.0, confidence=0.88),  # Disjoint
    ]
    kept = temporal_nms(proposals, iou_threshold=0.5)
    assert len(kept) == 2
    assert [p.event_id for p in kept] == ["P1", "P3"]
