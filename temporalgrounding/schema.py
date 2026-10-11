"""
Data Schemas and Contracts for Temporal Grounding Lab.
Multimodal Video & Audio Temporal Event Localization & Timestamp Hallucination Benchmark.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


class EventDomain(str, Enum):
    SURVEILLANCE_SECURITY = "Surveillance_Security"
    AUTONOMOUS_DRIVING = "Autonomous_Driving"
    MEDICAL_SURGICAL = "Medical_Surgical"
    SPORTS_ATHLETICS = "Sports_Athletics"
    PODCAST_DIALOGUE = "Podcast_Dialogue"


class ModelTier(str, Enum):
    UNIFORM_SAMPLE_BASELINE = "Uniform_Sample_Baseline"
    DENSE_FRAME_CAPTIONER = "Dense_Frame_Captioner"
    AUDIO_VISUAL_CROSS_ATTENTION = "Audio_Visual_Cross_Attention"
    TEMPORAL_SENTINEL = "TemporalGrounding_Sentinel"


@dataclass
class GroundTruthEvent:
    event_id: str
    label: str
    start_sec: float
    end_sec: float
    description: str
    modality: str = "audiovisual"  # "video", "audio", "audiovisual"

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "label": self.label,
            "start_sec": round(self.start_sec, 2),
            "end_sec": round(self.end_sec, 2),
            "duration_sec": round(self.duration_sec, 2),
            "description": self.description,
            "modality": self.modality,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroundTruthEvent":
        return cls(
            event_id=data["event_id"],
            label=data["label"],
            start_sec=float(data["start_sec"]),
            end_sec=float(data["end_sec"]),
            description=data.get("description", ""),
            modality=data.get("modality", "audiovisual"),
        )


@dataclass
class VideoScenario:
    video_id: str
    domain: EventDomain
    title: str
    duration_sec: float
    fps: float
    ground_truth_events: List[GroundTruthEvent]
    query_prompt: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "domain": self.domain.value,
            "title": self.title,
            "duration_sec": self.duration_sec,
            "fps": self.fps,
            "ground_truth_events": [e.to_dict() for e in self.ground_truth_events],
            "query_prompt": self.query_prompt,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VideoScenario":
        domain_raw = data.get("domain", "Surveillance_Security")
        try:
            domain = EventDomain(domain_raw)
        except ValueError:
            domain = EventDomain.SURVEILLANCE_SECURITY

        events = [GroundTruthEvent.from_dict(e) for e in data.get("ground_truth_events", [])]
        return cls(
            video_id=data["video_id"],
            domain=domain,
            title=data["title"],
            duration_sec=float(data["duration_sec"]),
            fps=float(data.get("fps", 30.0)),
            ground_truth_events=events,
            query_prompt=data.get("query_prompt", "Locate all key events in this video"),
        )


@dataclass
class PredictedInterval:
    event_id: str
    label: str
    start_sec: float
    end_sec: float
    confidence: float = 1.0

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "label": self.label,
            "start_sec": round(self.start_sec, 2),
            "end_sec": round(self.end_sec, 2),
            "duration_sec": round(self.duration_sec, 2),
            "confidence": round(self.confidence, 3),
        }


@dataclass
class EventEvaluationResult:
    video_id: str
    event_id: str
    model_tier: ModelTier
    tiou: float
    is_hit_03: bool
    is_hit_05: bool
    is_hit_07: bool
    is_hallucination: bool
    start_error_sec: float
    end_error_sec: float
    jitter_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "event_id": self.event_id,
            "model_tier": self.model_tier.value,
            "tiou": round(self.tiou, 4),
            "is_hit_03": self.is_hit_03,
            "is_hit_05": self.is_hit_05,
            "is_hit_07": self.is_hit_07,
            "is_hallucination": self.is_hallucination,
            "start_error_sec": round(self.start_error_sec, 3),
            "end_error_sec": round(self.end_error_sec, 3),
            "jitter_sec": round(self.jitter_sec, 3),
        }
