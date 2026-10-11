"""
Publication-Quality Visualization Suite for Temporal Grounding Lab.
Generates research charts comparing temporal localization models.
"""

import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np


class TemporalGroundingVisualizer:
    """
    Renders research figures for temporal event localization benchmarks.
    """

    COLOR_PALETTE = {
        "Uniform_Sample_Baseline": "#e74c3c",          # Red
        "Dense_Frame_Captioner": "#e67e22",            # Orange
        "Audio_Visual_Cross_Attention": "#3498db",      # Blue
        "TemporalGrounding_Sentinel": "#2ecc71",        # Emerald Green
    }

    @classmethod
    def plot_recall_at_iou_thresholds(
        cls,
        metrics_by_tier: Dict[str, Dict[str, Any]],
        output_path: str,
    ) -> None:
        """
        Plots Recall@1 across IoU thresholds [0.3, 0.5, 0.7].
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

        thresholds = [0.3, 0.5, 0.7]
        x_indices = np.arange(len(thresholds))

        for tier_name, metrics in metrics_by_tier.items():
            r_vals = [
                metrics["recall_at_1_iou_03"],
                metrics["recall_at_1_iou_05"],
                metrics["recall_at_1_iou_07"],
            ]
            color = cls.COLOR_PALETTE.get(tier_name, "#95a5a6")
            clean_label = tier_name.replace('_', ' ')

            ax.plot(
                x_indices,
                r_vals,
                marker="o",
                linewidth=2.5,
                markersize=8,
                label=clean_label,
                color=color,
            )
            for idx, val in enumerate(r_vals):
                ax.annotate(
                    f"{val:.1f}%",
                    (x_indices[idx], val),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    fontsize=8,
                    fontweight="bold",
                )

        ax.set_xticks(x_indices)
        ax.set_xticklabels(["IoU ≥ 0.3", "IoU ≥ 0.5 (Standard)", "IoU ≥ 0.7 (Strict)"], fontsize=10, fontweight="medium")
        ax.set_ylabel("Recall@1 (%) [Higher is Better]", fontsize=11, fontweight="medium")
        ax.set_title("Temporal Event Localization Recall@1 Across IoU Thresholds", fontsize=13, fontweight="bold", pad=12)
        ax.set_ylim(0, 115)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower left", frameon=True, framealpha=0.9, fontsize=9)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    @classmethod
    def plot_domain_performance(
        cls,
        metrics_by_tier: Dict[str, Dict[str, Any]],
        output_path: str,
    ) -> None:
        """
        Plots grouped bar chart of Mean tIoU (%) across the 5 domains.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        tiers = list(metrics_by_tier.keys())
        first_tier = next(iter(metrics_by_tier.values()))
        domains = list(first_tier.get("domain_miou", {}).keys())

        if not domains:
            return

        x = np.arange(len(domains))
        width = 0.8 / len(tiers)

        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

        for i, tier in enumerate(tiers):
            dom_vals = [metrics_by_tier[tier]["domain_miou"].get(dom, 0.0) for dom in domains]
            offset = (i - len(tiers) / 2) * width + width / 2
            color = cls.COLOR_PALETTE.get(tier, "#95a5a6")
            bars = ax.bar(x + offset, dom_vals, width, label=tier.replace('_', ' '), color=color, edgecolor="#2c3e50", alpha=0.9)
            ax.bar_label(bars, fmt="%.0f%%", padding=2, fontsize=7)

        clean_dom_labels = [d.replace('_', '\n') for d in domains]
        ax.set_xticks(x)
        ax.set_xticklabels(clean_dom_labels, fontsize=9)
        ax.set_ylabel("Mean Temporal IoU (mIoU %) [Higher is Better]", fontsize=11, fontweight="medium")
        ax.set_title("Cross-Domain Temporal Localization Performance (mIoU)", fontsize=13, fontweight="bold", pad=12)
        ax.set_ylim(0, 115)
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=8.5)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    @classmethod
    def plot_jitter_and_hallucination(
        cls,
        metrics_by_tier: Dict[str, Dict[str, Any]],
        output_path: str,
    ) -> None:
        """
        Plots dual-panel comparison of Boundary Jitter Error (seconds) and Temporal Hallucination Rate (%).
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), dpi=300)

        tiers = list(metrics_by_tier.keys())
        short_names = [t.replace('TemporalGrounding_', '').replace('_', '\n') for t in tiers]
        colors = [cls.COLOR_PALETTE.get(t, "#95a5a6") for t in tiers]

        jitter_vals = [metrics_by_tier[t]["boundary_jitter_sec"] for t in tiers]
        thr_vals = [metrics_by_tier[t]["temporal_hallucination_rate"] for t in tiers]

        # Panel 1: Boundary Jitter (seconds)
        b1 = ax1.bar(short_names, jitter_vals, color=colors, edgecolor="#2c3e50", width=0.6)
        ax1.bar_label(b1, fmt="%.2fs", padding=3, fontsize=9, fontweight="bold")
        ax1.set_title("Mean Boundary Jitter Error (s)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Boundary Error (seconds) [Lower is Better]", fontsize=10)
        ax1.grid(axis="y", linestyle=":", alpha=0.6)

        # Panel 2: Temporal Hallucination Rate (%)
        b2 = ax2.bar(short_names, thr_vals, color=colors, edgecolor="#2c3e50", width=0.6)
        ax2.bar_label(b2, fmt="%.1f%%", padding=3, fontsize=9, fontweight="bold")
        ax2.set_title("Temporal Hallucination Rate (THR %)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Hallucination Rate (%) [0% Ideal]", fontsize=10)
        ax2.set_ylim(0, max(thr_vals) * 1.35 if max(thr_vals) > 0 else 100)
        ax2.grid(axis="y", linestyle=":", alpha=0.6)

        plt.suptitle("Temporal Precision & Hallucination Mitigation Audit", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    @classmethod
    def plot_duration_scaling(
        cls,
        metrics_by_tier: Dict[str, Dict[str, Any]],
        output_path: str,
    ) -> None:
        """
        Plots performance retention across short (<30s), medium (30-180s), and long (>180s) videos.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)

        first_tier = next(iter(metrics_by_tier.values()))
        dur_buckets = list(first_tier.get("duration_miou", {}).keys())

        if not dur_buckets:
            return

        x = np.arange(len(dur_buckets))
        width = 0.8 / len(metrics_by_tier)

        for i, (tier_name, metrics) in enumerate(metrics_by_tier.items()):
            vals = [metrics["duration_miou"].get(b, 0.0) for b in dur_buckets]
            offset = (i - len(metrics_by_tier) / 2) * width + width / 2
            color = cls.COLOR_PALETTE.get(tier_name, "#95a5a6")
            bars = ax.bar(x + offset, vals, width, label=tier_name.replace('_', ' '), color=color, edgecolor="#2c3e50", alpha=0.9)
            ax.bar_label(bars, fmt="%.0f%%", padding=2, fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels(dur_buckets, fontsize=10, fontweight="medium")
        ax.set_ylabel("Mean tIoU (%) [Higher is Better]", fontsize=11, fontweight="medium")
        ax.set_title("Temporal Grounding Retention vs Video Duration Scale", fontsize=13, fontweight="bold", pad=12)
        ax.set_ylim(0, 115)
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=8.5)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
