"""
Temporal Grounding Lab: Streamlit Timeline Inspector & Evaluation Studio.
Interactive multi-track timeline visualization, CVPR-standard metrics observatory,
and live multimodal video reasoning playground.
"""

import os
import sys
import json
import streamlit as st
import pandas as pd
import numpy as np

# Ensure local package is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Also try to import openrouter client from parent workspace if present
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, workspace_dir)

from temporalgrounding.schema import VideoScenario, ModelTier, EventDomain
from temporalgrounding.metrics import compute_tiou, compute_boundary_jitter
from temporalgrounding.pipeline import TemporalGroundingPipeline
from temporalgrounding.evaluator import TemporalGroundingEvaluator
from temporalgrounding.visualizer import TemporalGroundingVisualizer

openrouter_client = None
try:
    from src.openrouter_client import OpenRouterClient
    openrouter_client = OpenRouterClient()
except Exception:
    pass

st.set_page_config(
    page_title="Temporal Grounding Lab | Video-LLM Localization Studio",
    page_icon="⏱️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #3498db, #9b59b6, #2ecc71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #7f8c8d;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .track-gt {
        background-color: #27ae60;
        color: white;
        padding: 5px 10px;
        border-radius: 6px;
        margin: 2px;
        display: inline-block;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .track-pred {
        background-color: #2980b9;
        color: white;
        padding: 5px 10px;
        border-radius: 6px;
        margin: 2px;
        display: inline-block;
        font-size: 0.85rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_scenarios() -> list:
    data_path = os.path.join(os.path.dirname(__file__), "data", "video_scenarios.json")
    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [VideoScenario.from_dict(item) for item in data]
    return []


scenarios = load_scenarios()

# Sidebar
with st.sidebar:
    st.markdown("### ⏱️ Temporal Grounding Control")
    st.markdown("**Video & Audio Localization Observatory**")
    st.markdown("---")

    mode = st.radio(
        "Execution Mode",
        ["High-Fidelity Simulation", "Live OpenRouter Free Model"],
        index=0,
    )

    selected_model = "openrouter/free"
    if mode == "Live OpenRouter Free Model":
        selected_model = st.selectbox(
            "OpenRouter Model",
            [
                "openrouter/free",
                "nvidia/nemotron-3-ultra-550b-a55b:free",
                "google/gemma-4-31b-it:free",
                "nvidia/nemotron-3.5-lightning:free",
            ],
            index=0,
        )

    st.markdown("---")
    st.markdown("### 📋 Benchmark Domains")
    st.markdown("- 🛡️ Surveillance & Security")
    st.markdown("- 🚗 Autonomous Driving")
    st.markdown("- 🏥 Medical & Surgical")
    st.markdown("- 🏀 Sports & Athletics")
    st.markdown("- 🎙️ Podcast & Dialogue")
    st.markdown("---")
    st.caption("Temporal Grounding Lab v1.0.0")

# Header
st.markdown('<div class="main-header">Temporal Grounding Lab: Video-LLM Localization Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-Modal Temporal Event Localization & Timestamp Hallucination Benchmark Observatory</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎬 Interactive Multi-Track Timeline Inspector",
    "📊 Benchmark Observatory & Metrics",
    "📁 Scenario Explorer (25 Episodes)",
    "🧪 Live Temporal Interval Auditor",
])

# --- TAB 1: INTERACTIVE MULTI-TRACK TIMELINE INSPECTOR ---
with tab1:
    st.subheader("Multi-Track Temporal Event Inspector")
    st.markdown("Inspect ground-truth vs model-predicted temporal boundaries across any video episode.")

    col_sc, col_model = st.columns([2, 1])
    with col_sc:
        sc_titles = {f"[{s.video_id}] {s.title} ({s.duration_sec:.0f}s - {s.domain.value})": s for s in scenarios}
        selected_sc_title = st.selectbox("Select Video Scenario", list(sc_titles.keys()))
        curr_scenario = sc_titles[selected_sc_title]

    with col_model:
        selected_tier_str = st.selectbox(
            "Select Model Architecture",
            [t.value for t in ModelTier],
            index=3,
        )
        curr_tier = ModelTier(selected_tier_str)

    st.markdown(f"**Query**: *\"{curr_scenario.query_prompt}\"*")
    st.caption(f"Domain: `{curr_scenario.domain.value}` | Total Duration: `{curr_scenario.duration_sec:.1f}s` | Events: `{len(curr_scenario.ground_truth_events)}`")

    pipeline = TemporalGroundingPipeline(
        openrouter_client=openrouter_client if mode == "Live OpenRouter Free Model" else None,
        model_name=selected_model,
    )
    predictions = pipeline.predict_scenario(curr_scenario, curr_tier)
    pred_map = {p.event_id: p for p in predictions}

    # Timeline Visualization
    st.markdown("### 🎞️ Visual Multi-Track Timeline")

    # Time scrubber
    scrub_time = st.slider("Timeline Scrubber (seconds)", 0.0, curr_scenario.duration_sec, 0.0, step=0.5)

    # Active events at scrub_time
    active_gt = [e for e in curr_scenario.ground_truth_events if e.start_sec <= scrub_time <= e.end_sec]
    active_pred = [p for p in predictions if p.start_sec <= scrub_time <= p.end_sec]

    col_scrub1, col_scrub2 = st.columns(2)
    with col_scrub1:
        st.markdown(f"**Ground Truth at {scrub_time:.1f}s**:")
        if active_gt:
            for ag in active_gt:
                st.success(f"🟢 **{ag.label}** ({ag.start_sec:.1f}s - {ag.end_sec:.1f}s)")
        else:
            st.caption("No event active.")

    with col_scrub2:
        st.markdown(f"**Predicted at {scrub_time:.1f}s**:")
        if active_pred:
            for ap in active_pred:
                st.info(f"🔵 **{ap.label}** ({ap.start_sec:.1f}s - {ap.end_sec:.1f}s | conf: {ap.confidence:.2f})")
        else:
            st.caption("No prediction active.")

    # Event Comparison Table
    st.markdown("### Granular Event Overlap & tIoU Breakdown")
    comparison_rows = []
    total_tiou = 0.0
    for gt in curr_scenario.ground_truth_events:
        p = pred_map.get(gt.event_id)
        if p:
            tiou = compute_tiou(p.start_sec, p.end_sec, gt.start_sec, gt.end_sec)
            jitter = compute_boundary_jitter(p.start_sec, p.end_sec, gt.start_sec, gt.end_sec)
            total_tiou += tiou
            comparison_rows.append({
                "Event Label": gt.label,
                "Modality": gt.modality,
                "Ground Truth Interval": f"[{gt.start_sec:.1f}s - {gt.end_sec:.1f}s] ({gt.duration_sec:.1f}s)",
                "Predicted Interval": f"[{p.start_sec:.1f}s - {p.end_sec:.1f}s] ({p.duration_sec:.1f}s)",
                "tIoU (%)": f"{tiou * 100.0:.1f}%",
                "Boundary Jitter": f"{jitter:.2f}s",
                "Status": "✅ HIT" if tiou >= 0.5 else "⚠️ MISS / DRIFT",
            })
        else:
            comparison_rows.append({
                "Event Label": gt.label,
                "Modality": gt.modality,
                "Ground Truth Interval": f"[{gt.start_sec:.1f}s - {gt.end_sec:.1f}s]",
                "Predicted Interval": "MISSED (No Detection)",
                "tIoU (%)": "0.0%",
                "Boundary Jitter": f"{curr_scenario.duration_sec:.1f}s",
                "Status": "❌ MISSED",
            })

    st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True)

    sc_miou = (total_tiou / len(curr_scenario.ground_truth_events)) * 100.0
    st.metric("Scenario Mean tIoU", f"{sc_miou:.1f}%")

# --- TAB 2: BENCHMARK OBSERVATORY ---
with tab2:
    st.subheader("CVPR-Standard Benchmark Observatory (25 Scenarios)")
    st.markdown("Comprehensive evaluation across all 4 architectures.")

    evaluator = TemporalGroundingEvaluator()
    benchmark_data = evaluator.run_benchmark(scenarios)
    metrics_by_tier = benchmark_data["metrics_by_tier"]

    sentinel_m = metrics_by_tier[ModelTier.TEMPORAL_SENTINEL.value]
    baseline_m = metrics_by_tier[ModelTier.UNIFORM_SAMPLE_BASELINE.value]

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Sentinel mIoU", f"{sentinel_m['mean_tiou']:.1f}%", f"+{sentinel_m['mean_tiou'] - baseline_m['mean_tiou']:.1f}% vs Baseline")
    with m_col2:
        st.metric("Recall@1 (IoU≥0.5)", f"{sentinel_m['recall_at_1_iou_05']:.1f}%", f"+{sentinel_m['recall_at_1_iou_05'] - baseline_m['recall_at_1_iou_05']:.1f}%")
    with m_col3:
        st.metric("Temporal Hallucination", f"{sentinel_m['temporal_hallucination_rate']:.1f}%", f"-{baseline_m['temporal_hallucination_rate'] - sentinel_m['temporal_hallucination_rate']:.1f}%")
    with m_col4:
        st.metric("Boundary Jitter Error", f"{sentinel_m['boundary_jitter_sec']:.2f}s", f"-{baseline_m['boundary_jitter_sec'] - sentinel_m['boundary_jitter_sec']:.2f}s")

    st.markdown("### Multi-Model Quantitative Matrix")
    df_metrics = []
    for tier, m in metrics_by_tier.items():
        df_metrics.append({
            "Architecture": tier.replace('_', ' '),
            "mIoU (%) ↑": f"{m['mean_tiou']:.1f}%",
            "R@1 (0.3) ↑": f"{m['recall_at_1_iou_03']:.1f}%",
            "R@1 (0.5) ↑": f"{m['recall_at_1_iou_05']:.1f}%",
            "R@1 (0.7) ↑": f"{m['recall_at_1_iou_07']:.1f}%",
            "THR (%) ↓": f"{m['temporal_hallucination_rate']:.1f}%",
            "Jitter (s) ↓": f"{m['boundary_jitter_sec']:.2f}s",
            "Ordering Acc ↑": f"{m['ordering_accuracy']:.1f}%",
            "Latency": f"{m['avg_latency_ms']:.2f} ms",
        })
    st.dataframe(pd.DataFrame(df_metrics), use_container_width=True)

    # Figures Section
    st.markdown("### Publication Figures")
    fig_col1, fig_col2 = st.columns(2)

    reports_dir = os.path.join(os.path.dirname(__file__), "reports", "figures")
    f1_path = os.path.join(reports_dir, "tiou_recall_curves.png")
    f2_path = os.path.join(reports_dir, "domain_performance_breakdown.png")
    f3_path = os.path.join(reports_dir, "temporal_jitter_and_hallucination.png")
    f4_path = os.path.join(reports_dir, "duration_scaling_drift.png")

    if not os.path.exists(f1_path):
        TemporalGroundingVisualizer.plot_recall_at_iou_thresholds(metrics_by_tier, f1_path)
        TemporalGroundingVisualizer.plot_domain_performance(metrics_by_tier, f2_path)
        TemporalGroundingVisualizer.plot_jitter_and_hallucination(metrics_by_tier, f3_path)
        TemporalGroundingVisualizer.plot_duration_scaling(metrics_by_tier, f4_path)

    with fig_col1:
        if os.path.exists(f1_path):
            st.image(f1_path, caption="Figure 1: Recall@1 Across IoU Thresholds")
        if os.path.exists(f3_path):
            st.image(f3_path, caption="Figure 3: Boundary Jitter Error & Hallucination Rate")

    with fig_col2:
        if os.path.exists(f2_path):
            st.image(f2_path, caption="Figure 2: Cross-Domain Performance (mIoU)")
        if os.path.exists(f4_path):
            st.image(f4_path, caption="Figure 4: Temporal Grounding Retention vs Duration Scale")

# --- TAB 3: SCENARIO EXPLORER ---
with tab3:
    st.subheader("Multimodal Dataset Explorer (25 Scenarios)")
    st.markdown("Browse all 25 curated episodes across 5 enterprise domains.")

    domain_filter = st.selectbox("Filter by Domain", ["ALL"] + [d.value for d in EventDomain])
    filtered_sc = scenarios
    if domain_filter != "ALL":
        filtered_sc = [s for s in scenarios if s.domain.value == domain_filter]

    st.markdown(f"**Displaying {len(filtered_sc)} video scenarios:**")

    for s in filtered_sc:
        with st.expander(f"[{s.video_id}] {s.title} ({s.duration_sec:.0f}s - {s.domain.value})"):
            st.markdown(f"**Query**: *\"{s.query_prompt}\"*")
            st.markdown(f"**Duration**: `{s.duration_sec} seconds` | FPS: `{s.fps}`")
            st.markdown("**Ground Truth Events**:")
            for ev in s.ground_truth_events:
                mod_icon = "🎬" if ev.modality == "video" else ("🔊" if ev.modality == "audio" else "🎥🔊")
                st.markdown(f"- {mod_icon} **{ev.label}**: `[{ev.start_sec:.1f}s - {ev.end_sec:.1f}s]` (Duration: `{ev.duration_sec:.1f}s`) — *{ev.description}*")

# --- TAB 4: LIVE TEMPORAL AUDITOR ---
with tab4:
    st.subheader("Temporal Interval Precision Auditor")
    st.markdown("Calculate exact 1D Temporal IoU (tIoU) and Boundary Jitter Error between any two intervals.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Ground Truth Event Interval")
        gt_s = st.number_input("GT Start (seconds)", value=14.5, step=0.1)
        gt_e = st.number_input("GT End (seconds)", value=26.0, step=0.1)

    with c2:
        st.markdown("#### Model Predicted Interval")
        pr_s = st.number_input("Pred Start (seconds)", value=15.0, step=0.1)
        pr_e = st.number_input("Pred End (seconds)", value=25.8, step=0.1)

    tiou_calc = compute_tiou(gt_s, gt_e, pr_s, pr_e)
    jitter_calc = compute_boundary_jitter(gt_s, gt_e, pr_s, pr_e)

    res_col1, res_col2, res_col3 = st.columns(3)
    with res_col1:
        st.metric("Temporal IoU (tIoU)", f"{tiou_calc * 100.0:.2f}%")
    with res_col2:
        st.metric("Boundary Jitter", f"{jitter_calc:.2f}s")
    with res_col3:
        status_text = "PASSED (IoU ≥ 0.5)" if tiou_calc >= 0.5 else "FAILED (IoU < 0.5)"
        st.metric("Validation Status", status_text)
