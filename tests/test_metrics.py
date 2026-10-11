"""
Unit tests for Temporal Grounding metrics.
"""

import pytest
from temporalgrounding.metrics import (
    compute_tiou,
    compute_boundary_jitter,
    check_hallucination,
    compute_temporal_ordering_accuracy,
)
from temporalgrounding.schema import GroundTruthEvent, PredictedInterval


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
