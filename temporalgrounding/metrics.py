"""
Temporal Grounding Metrics Engine.
Computes 1D Temporal IoU (tIoU), Recall@K at thresholds (0.3, 0.5, 0.7),
Boundary Jitter Error (BJE), Temporal Hallucination Rate (THR), and Temporal Ordering Accuracy (TOA).
"""

from typing import List, Dict, Any, Tuple, Optional
from temporalgrounding.schema import (
    GroundTruthEvent,
    PredictedInterval,
    EventEvaluationResult,
    ModelTier,
    VideoScenario,
)


def compute_tiou(start1: float, end1: float, start2: float, end2: float) -> float:
    """
    Computes 1D Temporal Intersection-over-Union (tIoU) between two time intervals.
    """
    s1, e1 = min(start1, end1), max(start1, end1)
    s2, e2 = min(start2, end2), max(start2, end2)

    intersection = max(0.0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)

    if union <= 0.0:
        return 0.0
    return max(0.0, min(1.0, intersection / union))


def compute_boundary_jitter(start_pred: float, end_pred: float, start_gt: float, end_gt: float) -> float:
    """
    Computes mean absolute boundary jitter error in seconds:
    (|start_pred - start_gt| + |end_pred - end_gt|) / 2.0
    """
    start_err = abs(start_pred - start_gt)
    end_err = abs(end_pred - end_gt)
    return (start_err + end_err) / 2.0


def check_hallucination(
    tiou: float,
    start_pred: float,
    end_pred: float,
    video_duration: float,
) -> bool:
    """
    Determines if a predicted interval represents a temporal hallucination:
    - Zero overlap (tIoU = 0.0) with ground truth
    - Prediction outside video duration boundaries
    - Inverted temporal boundaries
    """
    if tiou <= 0.0:
        return True
    if start_pred < 0.0 or start_pred >= video_duration:
        return True
    if end_pred < start_pred:
        return True
    return False


def compute_temporal_ordering_accuracy(
    predicted_events: List[PredictedInterval],
    gt_events: List[GroundTruthEvent],
) -> float:
    """
    Evaluates temporal sequence causality.
    Computes fraction of concordant pairwise event orderings:
    Pair (i, j) is concordant if (start_gt[i] < start_gt[j]) == (start_pred[i] < start_pred[j]).
    """
    if len(gt_events) < 2 or len(predicted_events) < 2:
        return 1.0

    pred_map = {p.event_id: p for p in predicted_events}
    valid_pairs = []

    for i in range(len(gt_events)):
        for j in range(i + 1, len(gt_events)):
            gt_i, gt_j = gt_events[i], gt_events[j]
            pred_i = pred_map.get(gt_i.event_id)
            pred_j = pred_map.get(gt_j.event_id)
            if pred_i and pred_j:
                valid_pairs.append((gt_i, gt_j, pred_i, pred_j))

    if not valid_pairs:
        return 0.0

    concordant = 0
    for gt_i, gt_j, pred_i, pred_j in valid_pairs:
        gt_order = gt_i.start_sec < gt_j.start_sec
        pred_order = pred_i.start_sec < pred_j.start_sec
        if gt_order == pred_order:
            concordant += 1

    return concordant / len(valid_pairs)


def evaluate_event_prediction(
    pred: Optional[PredictedInterval],
    gt: GroundTruthEvent,
    video_duration: float,
    model_tier: ModelTier,
) -> EventEvaluationResult:
    """
    Evaluates a single predicted interval against a ground truth event.
    """
    if pred is None:
        # Complete missed detection
        return EventEvaluationResult(
            video_id="",
            event_id=gt.event_id,
            model_tier=model_tier,
            tiou=0.0,
            is_hit_03=False,
            is_hit_05=False,
            is_hit_07=False,
            is_hallucination=True,
            start_error_sec=video_duration,
            end_error_sec=video_duration,
            jitter_sec=video_duration,
        )

    tiou = compute_tiou(pred.start_sec, pred.end_sec, gt.start_sec, gt.end_sec)
    jitter = compute_boundary_jitter(pred.start_sec, pred.end_sec, gt.start_sec, gt.end_sec)
    is_halluc = check_hallucination(tiou, pred.start_sec, pred.end_sec, video_duration)

    return EventEvaluationResult(
        video_id="",
        event_id=gt.event_id,
        model_tier=model_tier,
        tiou=tiou,
        is_hit_03=(tiou >= 0.3),
        is_hit_05=(tiou >= 0.5),
        is_hit_07=(tiou >= 0.7),
        is_hallucination=is_halluc,
        start_error_sec=abs(pred.start_sec - gt.start_sec),
        end_error_sec=abs(pred.end_sec - gt.end_sec),
        jitter_sec=jitter,
    )
