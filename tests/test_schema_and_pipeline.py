"""
Unit tests for schema contracts and temporal grounding pipeline.
"""

import pytest
from temporalgrounding.schema import (
    EventDomain,
    ModelTier,
    GroundTruthEvent,
    VideoScenario,
)
from temporalgrounding.pipeline import TemporalGroundingPipeline


@pytest.fixture
def mock_scenario():
    return VideoScenario(
        video_id="TEST-01",
        domain=EventDomain.SURVEILLANCE_SECURITY,
        title="Test Fence Incident",
        duration_sec=60.0,
        fps=30.0,
        query_prompt="Find the intrusion",
        ground_truth_events=[
            GroundTruthEvent(
                event_id="TEST-01-E1",
                label="Approach",
                start_sec=10.0,
                end_sec=20.0,
                description="Subject approaches",
                modality="video",
            ),
            GroundTruthEvent(
                event_id="TEST-01-E2",
                label="Cut Wire",
                start_sec=25.0,
                end_sec=30.0,
                description="Fence snips",
                modality="audiovisual",
            ),
        ],
    )


def test_schema_serialization(mock_scenario):
    data = mock_scenario.to_dict()
    restored = VideoScenario.from_dict(data)
    assert restored.video_id == "TEST-01"
    assert len(restored.ground_truth_events) == 2
    assert restored.ground_truth_events[0].duration_sec == 10.0


def test_pipeline_prediction_all_tiers(mock_scenario):
    pipeline = TemporalGroundingPipeline()
    for tier in ModelTier:
        preds = pipeline.predict_scenario(mock_scenario, tier)
        assert len(preds) <= 2
        for p in preds:
            assert p.start_sec >= 0.0
            assert p.end_sec <= mock_scenario.duration_sec
            assert p.end_sec >= p.start_sec
