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
    "TemporalGroundingPipeline",
    "TemporalGroundingEvaluator",
    "TemporalGroundingVisualizer",
    "TemporalGroundingReportGenerator",
]
