"""
Temporal Grounding Lab: Multimodal Video & Audio Event Localization Package.
"""

from temporalgrounding.schema import (
    EventDomain,
    ModelTier,
    GroundTruthEvent,
    VideoScenario,
    PredictedInterval,
    EventEvaluationResult,
)
from temporalgrounding.metrics import (
    compute_tiou,
    compute_boundary_jitter,
    check_hallucination,
    compute_temporal_ordering_accuracy,
    evaluate_event_prediction,
    compute_recall_auc,
    diagnose_boundary_bias,
    temporal_nms,
)
from temporalgrounding.pipeline import TemporalGroundingPipeline
from temporalgrounding.evaluator import TemporalGroundingEvaluator
from temporalgrounding.visualizer import TemporalGroundingVisualizer
from temporalgrounding.report_generator import TemporalGroundingReportGenerator

__all__ = [
    "EventDomain",
    "ModelTier",
    "GroundTruthEvent",
    "VideoScenario",
    "PredictedInterval",
    "EventEvaluationResult",
    "compute_tiou",
    "compute_boundary_jitter",
    "check_hallucination",
    "compute_temporal_ordering_accuracy",
    "evaluate_event_prediction",
    "compute_recall_auc",
    "diagnose_boundary_bias",
    "temporal_nms",
    "TemporalGroundingPipeline",
    "TemporalGroundingEvaluator",
    "TemporalGroundingVisualizer",
    "TemporalGroundingReportGenerator",
]
