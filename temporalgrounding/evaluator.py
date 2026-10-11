"""
Benchmark Evaluator and Aggregator for Temporal Grounding Lab.
Evaluates multi-model performance across video scenarios and computes CVPR-standard metrics.
"""

import time
from typing import List, Dict, Any, Optional
from temporalgrounding.schema import (
    ModelTier,
    VideoScenario,
    GroundTruthEvent,
    PredictedInterval,
    EventEvaluationResult,
    EventDomain,
)
from temporalgrounding.metrics import (
    evaluate_event_prediction,
    compute_temporal_ordering_accuracy,
)
from temporalgrounding.pipeline import TemporalGroundingPipeline


class TemporalGroundingEvaluator:
    """
    Evaluator engine aggregating scenario- and domain-level temporal metrics.
    """

    def __init__(self, pipeline: Optional[TemporalGroundingPipeline] = None):
        self.pipeline = pipeline or TemporalGroundingPipeline()

    def run_benchmark(
        self,
        scenarios: List[VideoScenario],
        tiers: Optional[List[ModelTier]] = None,
    ) -> Dict[str, Any]:
        """Runs evaluation across all scenarios and model tiers."""
        if tiers is None:
            tiers = [
                ModelTier.UNIFORM_SAMPLE_BASELINE,
                ModelTier.DENSE_FRAME_CAPTIONER,
                ModelTier.AUDIO_VISUAL_CROSS_ATTENTION,
                ModelTier.TEMPORAL_SENTINEL,
            ]

        results_by_tier: Dict[str, List[EventEvaluationResult]] = {t.value: [] for t in tiers}
        scenario_results_by_tier: Dict[str, List[Dict[str, Any]]] = {t.value: [] for t in tiers}

        for tier in tiers:
            for sc in scenarios:
                t0 = time.perf_counter()
                preds = self.pipeline.predict_scenario(sc, tier)
                latency_ms = (time.perf_counter() - t0) * 1000.0

                pred_map = {p.event_id: p for p in preds}

                event_evals = []
                for gt in sc.ground_truth_events:
                    p = pred_map.get(gt.event_id)
                    eval_res = evaluate_event_prediction(p, gt, sc.duration_sec, tier)
                    eval_res.video_id = sc.video_id
                    event_evals.append(eval_res)
                    results_by_tier[tier.value].append(eval_res)

                # Scenario level metrics
                ordering_acc = compute_temporal_ordering_accuracy(preds, sc.ground_truth_events)
                sc_miou = (sum(e.tiou for e in event_evals) / len(event_evals)) if event_evals else 0.0

                scenario_results_by_tier[tier.value].append({
                    "video_id": sc.video_id,
                    "domain": sc.domain.value,
                    "duration_sec": sc.duration_sec,
                    "events_count": len(sc.ground_truth_events),
                    "mean_tiou": round(sc_miou, 4),
                    "ordering_accuracy": round(ordering_acc * 100.0, 1),
                    "latency_ms": round(latency_ms, 2),
                })

        # Compute tier summary metrics
        metrics_by_tier: Dict[str, Dict[str, Any]] = {}
        for tier in tiers:
            tier_key = tier.value
            event_results = results_by_tier[tier_key]
            sc_results = scenario_results_by_tier[tier_key]
            metrics_by_tier[tier_key] = self.compute_tier_metrics(event_results, sc_results, scenarios, tier)

        return {
            "total_scenarios": len(scenarios),
            "total_events": sum(len(sc.ground_truth_events) for sc in scenarios),
            "tiers_evaluated": [t.value for t in tiers],
            "metrics_by_tier": metrics_by_tier,
            "raw_event_results": results_by_tier,
            "scenario_results": scenario_results_by_tier,
        }

    @staticmethod
    def compute_tier_metrics(
        event_results: List[EventEvaluationResult],
        sc_results: List[Dict[str, Any]],
        scenarios: List[VideoScenario],
        tier: ModelTier,
    ) -> Dict[str, Any]:
        """Computes aggregated metrics for a single model tier."""
        total_events = len(event_results)
        if total_events == 0:
            return {}

        total_tiou = sum(e.tiou for e in event_results)
        hit_03 = sum(1 for e in event_results if e.is_hit_03)
        hit_05 = sum(1 for e in event_results if e.is_hit_05)
        hit_07 = sum(1 for e in event_results if e.is_hit_07)
        hallucinations = sum(1 for e in event_results if e.is_hallucination)
        total_jitter = sum(e.jitter_sec for e in event_results)

        miou = total_tiou / total_events
        r_at_03 = (hit_03 / total_events) * 100.0
        r_at_05 = (hit_05 / total_events) * 100.0
        r_at_07 = (hit_07 / total_events) * 100.0
        thr = (hallucinations / total_events) * 100.0
        mean_jitter = total_jitter / total_events

        mean_ordering = (sum(s["ordering_accuracy"] for s in sc_results) / len(sc_results)) if sc_results else 0.0
        mean_latency = (sum(s["latency_ms"] for s in sc_results) / len(sc_results)) if sc_results else 0.0

        # Domain breakdown
        sc_domain_map = {sc.video_id: sc.domain.value for sc in scenarios}
        domain_tiou: Dict[str, List[float]] = {}
        for e in event_results:
            dom = sc_domain_map.get(e.video_id, "Unknown")
            domain_tiou.setdefault(dom, []).append(e.tiou)

        domain_miou = {dom: round((sum(vals) / len(vals)) * 100.0, 1) for dom, vals in domain_tiou.items()}

        # Duration scaling breakdown
        sc_dur_map = {sc.video_id: sc.duration_sec for sc in scenarios}
        dur_buckets = {"short (<30s)": [], "medium (30-180s)": [], "long (>180s)": []}
        for e in event_results:
            d = sc_dur_map.get(e.video_id, 60.0)
            if d < 30.0:
                dur_buckets["short (<30s)"].append(e.tiou)
            elif d <= 180.0:
                dur_buckets["medium (30-180s)"].append(e.tiou)
            else:
                dur_buckets["long (>180s)"].append(e.tiou)

        duration_miou = {
            b: round((sum(v) / len(v)) * 100.0, 1) if v else 0.0
            for b, v in dur_buckets.items()
        }

        return {
            "model_tier": tier.value,
            "mean_tiou": round(miou * 100.0, 1),
            "recall_at_1_iou_03": round(r_at_03, 1),
            "recall_at_1_iou_05": round(r_at_05, 1),
            "recall_at_1_iou_07": round(r_at_07, 1),
            "temporal_hallucination_rate": round(thr, 1),
            "boundary_jitter_sec": round(mean_jitter, 2),
            "ordering_accuracy": round(mean_ordering, 1),
            "avg_latency_ms": round(mean_latency, 2),
            "domain_miou": domain_miou,
            "duration_miou": duration_miou,
        }
