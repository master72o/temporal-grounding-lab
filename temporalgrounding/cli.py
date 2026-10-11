"""
Temporal Grounding Lab Command-Line Interface (CLI).
Automates benchmarking, single-video evaluation, and interval auditing.
"""

import os
import sys
import json
import argparse
from typing import List

# Ensure package directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from temporalgrounding.schema import VideoScenario, ModelTier
from temporalgrounding.evaluator import TemporalGroundingEvaluator
from temporalgrounding.visualizer import TemporalGroundingVisualizer
from temporalgrounding.report_generator import TemporalGroundingReportGenerator
from temporalgrounding.metrics import compute_tiou, compute_boundary_jitter


def load_scenarios(path: str) -> List[VideoScenario]:
    """Loads video scenarios from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [VideoScenario.from_dict(item) for item in data]


def run_benchmark_cmd(args):
    """Executes multi-model temporal benchmark."""
    data_path = args.dataset or os.path.join(os.path.dirname(__file__), "..", "data", "video_scenarios.json")
    data_path = os.path.abspath(data_path)

    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        sys.exit(1)

    print(f"[*] Loading Temporal Grounding Benchmark from: {data_path}")
    scenarios = load_scenarios(data_path)
    print(f"[*] Loaded {len(scenarios)} video scenarios across 5 domains ({sum(len(s.ground_truth_events) for s in scenarios)} events).")

    evaluator = TemporalGroundingEvaluator()
    print("[*] Running multi-model temporal localization benchmark...")
    benchmark_data = evaluator.run_benchmark(scenarios)

    # Output paths
    output_dir = args.output_dir or os.path.join(os.path.dirname(__file__), "..", "reports")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, "figures")

    # Generate Reports
    md_path = os.path.join(output_dir, "temporal_grounding_report.md")
    json_path = os.path.join(output_dir, "benchmark_summary.json")
    csv_path = os.path.join(output_dir, "predictions_per_event.csv")

    TemporalGroundingReportGenerator.generate_markdown_report(benchmark_data, md_path)
    TemporalGroundingReportGenerator.export_json(benchmark_data, json_path)
    TemporalGroundingReportGenerator.export_csv(benchmark_data["raw_event_results"], csv_path)
    print(f"[+] Exported Markdown report: {md_path}")
    print(f"[+] Exported JSON metrics: {json_path}")
    print(f"[+] Exported CSV traces: {csv_path}")

    # Generate Figures
    if not args.no_figures:
        f1 = os.path.join(figures_dir, "tiou_recall_curves.png")
        f2 = os.path.join(figures_dir, "domain_performance_breakdown.png")
        f3 = os.path.join(figures_dir, "temporal_jitter_and_hallucination.png")
        f4 = os.path.join(figures_dir, "duration_scaling_drift.png")

        metrics_by_tier = benchmark_data["metrics_by_tier"]
        TemporalGroundingVisualizer.plot_recall_at_iou_thresholds(metrics_by_tier, f1)
        TemporalGroundingVisualizer.plot_domain_performance(metrics_by_tier, f2)
        TemporalGroundingVisualizer.plot_jitter_and_hallucination(metrics_by_tier, f3)
        TemporalGroundingVisualizer.plot_duration_scaling(metrics_by_tier, f4)
        print(f"[+] Generated publication figures in: {figures_dir}")

    # Display console summary table
    print("\n" + "=" * 95)
    print("TEMPORAL GROUNDING BENCHMARK RESULTS")
    print("=" * 95)
    print(f"{'Model Architecture':<30} | {'mIoU (%)':<9} | {'R@1 (0.5)':<10} | {'R@1 (0.7)':<10} | {'THR (%)':<8} | {'Jitter (s)':<10}")
    print("-" * 95)
    for tier, m in benchmark_data["metrics_by_tier"].items():
        print(f"{tier:<30} | {m['mean_tiou']:<9.1f} | {m['recall_at_1_iou_05']:<10.1f} | {m['recall_at_1_iou_07']:<10.1f} | {m['temporal_hallucination_rate']:<8.1f} | {m['boundary_jitter_sec']:<10.2f}")
    print("=" * 95 + "\n")


def audit_interval_cmd(args):
    """Audits two intervals."""
    tiou = compute_tiou(args.s1, args.e1, args.s2, args.e2)
    jitter = compute_boundary_jitter(args.s1, args.e1, args.s2, args.e2)
    print(f"\n[*] Interval 1 (Ground Truth): [{args.s1:.2f}s - {args.e1:.2f}s] (Duration: {args.e1 - args.s1:.2f}s)")
    print(f"[*] Interval 2 (Predicted):    [{args.s2:.2f}s - {args.e2:.2f}s] (Duration: {args.e2 - args.s2:.2f}s)")
    print("-" * 60)
    print(f"Temporal IoU (tIoU):      {tiou * 100.0:.2f}%")
    print(f"Boundary Jitter Error:    {jitter:.2f} seconds")
    print(f"Recall@1 (IoU >= 0.5):    {'PASS' if tiou >= 0.5 else 'FAIL'}")
    print(f"Recall@1 (IoU >= 0.7):    {'PASS' if tiou >= 0.7 else 'FAIL'}")
    print("-" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Temporal Grounding Lab CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # benchmark
    b_parser = subparsers.add_parser("benchmark", help="Run full benchmark")
    b_parser.add_argument("--dataset", type=str, default=None, help="Path to video_scenarios.json")
    b_parser.add_argument("--output-dir", type=str, default=None, help="Directory to save reports")
    b_parser.add_argument("--no-figures", action="store_true", help="Skip figure generation")

    # audit
    a_parser = subparsers.add_parser("audit", help="Audit two temporal intervals")
    a_parser.add_argument("--s1", type=float, required=True, help="Start 1 (seconds)")
    a_parser.add_argument("--e1", type=float, required=True, help="End 1 (seconds)")
    a_parser.add_argument("--s2", type=float, required=True, help="Start 2 (seconds)")
    a_parser.add_argument("--e2", type=float, required=True, help="End 2 (seconds)")

    args = parser.parse_args()
    if args.command == "benchmark":
        run_benchmark_cmd(args)
    elif args.command == "audit":
        audit_interval_cmd(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
