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


def compute_recall_auc(
    event_results: List[EventEvaluationResult],
    thresholds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Computes the Recall@1 curve across multiple tIoU thresholds (default 0.1 to 0.9)
    and integrates the Area Under the Recall-vs-tIoU Curve (R@1-AUC) via trapezoidal rule.
    """
    if thresholds is None:
        thresholds = [round(0.1 * i, 1) for i in range(1, 10)]
    if not event_results or not thresholds:
        return {"auc": 0.0, "curve": {}}

    n = len(event_results)
    curve: Dict[float, float] = {}
    recalls: List[float] = []
    for tau in sorted(thresholds):
        rec = sum(1 for e in event_results if e.tiou >= tau) / n
        curve[tau] = round(rec, 4)
        recalls.append(rec)

    if len(thresholds) < 2:
        auc = recalls[0]
    else:
        sorted_t = sorted(thresholds)
        span = sorted_t[-1] - sorted_t[0]
        trapz_sum = sum(
            0.5 * (recalls[i] + recalls[i + 1]) * (sorted_t[i + 1] - sorted_t[i])
            for i in range(len(sorted_t) - 1)
        )
        auc = trapz_sum / span if span > 0 else recalls[0]

    return {"auc": round(float(auc), 4), "curve": curve}


def diagnose_boundary_bias(
    pred_start: float,
    pred_end: float,
    gt_start: float,
    gt_end: float,
    ratio_tol: float = 0.25,
) -> str:
    """
    Classifies temporal boundary error modes into diagnostic categories:
    - 'hallucinated_disjoint': zero overlap with ground truth
    - 'over_extended': predicted duration is > (1 + ratio_tol) of ground truth
    - 'under_extended': predicted duration is < (1 - ratio_tol) of ground truth
    - 'early_shift': center shifted significantly earlier than ground truth
    - 'late_shift': center shifted significantly later than ground truth
    - 'tight_localization': high-precision boundary alignment
    """
    tiou = compute_tiou(pred_start, pred_end, gt_start, gt_end)
    if tiou <= 0.0:
        return "hallucinated_disjoint"

    gt_dur = max(1e-3, gt_end - gt_start)
    pred_dur = max(0.0, pred_end - pred_start)
    dur_ratio = pred_dur / gt_dur

    if dur_ratio > (1.0 + ratio_tol):
        return "over_extended"
    if dur_ratio < (1.0 - ratio_tol):
        return "under_extended"

    gt_center = 0.5 * (gt_start + gt_end)
    pred_center = 0.5 * (pred_start + pred_end)
    center_shift = (pred_center - gt_center) / gt_dur

    if center_shift < -0.15:
        return "early_shift"
    if center_shift > 0.15:
        return "late_shift"
    return "tight_localization"


def temporal_nms(
    intervals: List[PredictedInterval],
    iou_threshold: float = 0.5,
) -> List[PredictedInterval]:
    """
    1D Temporal Non-Maximum Suppression (NMS).
    Deduplicates overlapping candidate event proposals in descending confidence order.
    """
    if not intervals:
        return []

    sorted_candidates = sorted(intervals, key=lambda x: x.confidence, reverse=True)
    kept: List[PredictedInterval] = []

    while sorted_candidates:
        best = sorted_candidates.pop(0)
        kept.append(best)
        remaining = []
        for cand in sorted_candidates:
            same_label = (not best.label or not cand.label or best.label == cand.label)
            overlap = compute_tiou(best.start_sec, best.end_sec, cand.start_sec, cand.end_sec)
            if same_label and overlap >= iou_threshold:
                continue
            remaining.append(cand)
        sorted_candidates = remaining

    return sorted(kept, key=lambda x: x.start_sec)
