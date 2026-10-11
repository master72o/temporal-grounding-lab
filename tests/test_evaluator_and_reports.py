"""
Unit tests for evaluator and report generation.
"""

import os
import json
import pytest
from temporalgrounding.schema import (
    EventDomain,
    ModelTier,
    GroundTruthEvent,
    VideoScenario,
)
from temporalgrounding.evaluator import TemporalGroundingEvaluator
from temporalgrounding.report_generator import TemporalGroundingReportGenerator


@pytest.fixture
def test_scenarios():
    return [
        VideoScenario(
            video_id="TEST-01",
            domain=EventDomain.SURVEILLANCE_SECURITY,
            title="Surveillance Episode",
            duration_sec=60.0,
            fps=30.0,
            query_prompt="Locate breach",
            ground_truth_events=[
                GroundTruthEvent(
                    event_id="TEST-01-E1",
                    label="Breach",
                    start_sec=15.0,
                    end_sec=25.0,
                    description="Breach action",
                )
            ],
        ),
        VideoScenario(
            video_id="TEST-02",
            domain=EventDomain.AUTONOMOUS_DRIVING,
            title="Driving Episode",
            duration_sec=40.0,
            fps=30.0,
            query_prompt="Locate braking",
            ground_truth_events=[
                GroundTruthEvent(
                    event_id="TEST-02-E1",
                    label="Brake",
                    start_sec=10.0,
                    end_sec=16.0,
                    description="Emergency stop",
                )
            ],
        ),
    ]


def test_evaluator_and_reports(test_scenarios, tmp_path):
    evaluator = TemporalGroundingEvaluator()
    data = evaluator.run_benchmark(test_scenarios)

    assert data["total_scenarios"] == 2
    assert data["total_events"] == 2
    assert ModelTier.TEMPORAL_SENTINEL.value in data["metrics_by_tier"]

    sentinel_m = data["metrics_by_tier"][ModelTier.TEMPORAL_SENTINEL.value]
    assert sentinel_m["mean_tiou"] > 70.0
    assert sentinel_m["recall_at_1_iou_05"] >= 50.0

    md_file = str(tmp_path / "report.md")
    json_file = str(tmp_path / "summary.json")
    csv_file = str(tmp_path / "events.csv")

    TemporalGroundingReportGenerator.generate_markdown_report(data, md_file)
    TemporalGroundingReportGenerator.export_json(data, json_file)
    TemporalGroundingReportGenerator.export_csv(data["raw_event_results"], csv_file)

    assert os.path.exists(md_file)
    assert os.path.exists(json_file)
    assert os.path.exists(csv_file)

    with open(json_file, "r") as f:
        loaded = json.load(f)
    assert "metrics_by_tier" in loaded
