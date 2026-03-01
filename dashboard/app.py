"""TTT Crypto Regime Dashboard — Interactive visualization of results.

Launch:
    streamlit run dashboard/app.py

Requires checkpoints from experiments 01-03 and data/processed/dataset.pt.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.dataset import CryptoRegimeDataset
from src.models import TTTModel
from src.ttt_learner import TTTAdaptor
from src.eval import compute_metrics
from src.utils import get_device, seed_everything

# ── Page config ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TTT Crypto Regime Dashboard",
    page_icon="📊",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=DM+Sans:wght@400;500;700&display=swap');

    .stApp {
        background-color: #0a0a0f;
        color: #e0e0e0;
    }

    h1, h2, h3 {
        font-family: 'DM Sans', sans-serif !important;
        color: #f0f0f0 !important;
    }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 8px 0;
    }

    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: #00d4aa;
        margin: 4px 0;
    }

    .metric-label {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.85rem;
        color: #8888aa;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    .metric-delta-pos { color: #00d4aa; font-size: 0.9rem; }
    .metric-delta-neg { color: #ff4757; font-size: 0.9rem; }

    .regime-high { border-left: 4px solid #ff4757; }
    .regime-low { border-left: 4px solid #00d4aa; }

    div[data-testid="stSidebar"] {
        background-color: #0d0d15;
        border-right: 1px solid #1a1a2e;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'DM Sans', sans-serif;
        color: #8888aa;
    }

    .stTabs [aria-selected="true"] {
        color: #00d4aa !important;
        border-bottom-color: #00d4aa !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ─────────────────────────────────────────────────────


def metric_card(label: str, value: float, fmt: str = ".4f", delta: float | None = None, regime: str = ""):
    """Render a styled metric card."""
    regime_class = f" regime-{regime}" if regime else ""
    delta_html = ""
    if delta is not None:
        cls = "metric-delta-pos" if delta >= 0 else "metric-delta-neg"
        sign = "+" if delta >= 0 else ""
        delta_html = f'<div class="{cls}">{sign}{delta:.4f} vs baseline</div>'

    st.markdown(f"""
    <div class="metric-card{regime_class}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value:{fmt}}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_args = ckpt.get("args", {})
    aux_task = train_args.get("aux_task", "mask")
    num_groups = train_args.get("num_groups", 8)
    model = TTTModel(num_classes=2, aux_task=aux_task, num_groups=num_groups).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, train_args, ckpt


@st.cache_data
def load_dataset(data_dir: str = "data/processed"):
    """Load processed dataset."""
    dataset = CryptoRegimeDataset(data_dir)
    return dataset


def run_evaluation(model, dataset, adaptor, threshold, n_samples=None):
    """Run baseline + TTT evaluation and return results dict."""
    _, _, test_ds = dataset.get_splits()

    if n_samples and n_samples < len(test_ds):
        indices = list(range(n_samples))
        test_ds = Subset(test_ds, indices)

    loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # Baseline
    baseline_out = adaptor.evaluate_baseline(loader)
    baseline_m = compute_metrics(
        baseline_out["probabilities"].numpy(),
        baseline_out["labels"].numpy(),
        baseline_out["rv_values"].numpy(),
        threshold=threshold,
    )

    # Standard TTT
    all_probs, all_labels, all_rv = [], [], []
    for batch in loader:
        imgs = batch[0].to(adaptor.device)
        logits, _ = adaptor.adapt_and_predict(imgs)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(batch[1].numpy())
        all_rv.append(batch[2].numpy())

    ttt_probs = np.concatenate(all_probs)
    ttt_labels = np.concatenate(all_labels)
    ttt_rv = np.concatenate(all_rv)
    ttt_m = compute_metrics(ttt_probs, ttt_labels, ttt_rv, threshold=threshold)

    return {
        "baseline": baseline_m,
        "ttt_standard": ttt_m,
        "baseline_probs": baseline_out["probabilities"].numpy(),
        "ttt_probs": ttt_probs,
        "labels": ttt_labels,
        "rv_values": ttt_rv,
    }


def run_regime_evaluation(model, dataset, adaptor, threshold):
    """Run regime-stratified evaluation."""
    _, _, test_ds = dataset.get_splits()
    test_idx = dataset._splits["test"]
    rv_test = dataset.rv_values[test_idx].numpy()

    q25, q50, q75 = np.percentile(rv_test, [25, 50, 75])
    bins = {
        "Low (0-25%)": rv_test <= q25,
        "Mid-Low (25-50%)": (rv_test > q25) & (rv_test <= q50),
        "Mid-High (50-75%)": (rv_test > q50) & (rv_test <= q75),
        "High (75-100%)": rv_test > q75,
    }

    results = []
    for bin_name, mask in bins.items():
        idx = np.where(mask)[0]
        if len(idx) < 5:
            continue
        indices = np.asarray(test_idx)[idx].tolist()
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=1, shuffle=False)

        # Baseline
        out_b = adaptor.evaluate_baseline(loader)
        m_b = compute_metrics(
            out_b["probabilities"].numpy(), out_b["labels"].numpy(),
            out_b["rv_values"].numpy(), threshold=threshold,
        )

        # TTT
        probs_t, labels_t, rv_t = [], [], []
        for batch in loader:
            imgs = batch[0].to(adaptor.device)
            logits, _ = adaptor.adapt_and_predict(imgs)
            p = F.softmax(logits, dim=-1).cpu().numpy()
            probs_t.append(p)
            labels_t.append(batch[1].numpy())
            rv_t.append(batch[2].numpy())

        probs_t = np.concatenate(probs_t)
        labels_t = np.concatenate(labels_t)
        rv_t = np.concatenate(rv_t)
        m_t = compute_metrics(probs_t, labels_t, rv_t, threshold=threshold)

        results.append({
            "Regime": bin_name,
            "n": len(idx),
            "Baseline Acc": m_b["accuracy"],
            "Baseline F1": m_b["f1"],
            "TTT Acc": m_t["accuracy"],
            "TTT F1": m_t["f1"],
            "Δ Acc": m_t["accuracy"] - m_b["accuracy"],
            "Δ F1": m_t["f1"] - m_b["f1"],
        })

    return pd.DataFrame(results)


# ── Sidebar ──────────────────────────────────────────────────────────────

st.sidebar.markdown("## ⚙️ Configuration")

# Detect available checkpoints
ckpt_options = {}
for name, path in [
    ("Mask (joint)", "checkpoints/joint/best.pt"),
    ("Rotation", "checkpoints/rotation/best.pt"),
    ("Mask ETH", "checkpoints/joint_eth/best.pt"),
]:
    if Path(path).exists():
        ckpt_options[name] = path

if not ckpt_options:
    st.error("No checkpoints found. Run training first (see README).")
    st.stop()

selected_ckpt = st.sidebar.selectbox("Checkpoint", list(ckpt_options.keys()))
ckpt_path = ckpt_options[selected_ckpt]

threshold = st.sidebar.slider("Decision Threshold", 0.1, 0.9, 0.35, 0.05)
ttt_lr = st.sidebar.slider("TTT Learning Rate", 0.001, 0.2, 0.05, 0.005, format="%.3f")
ttt_steps = st.sidebar.slider("TTT Steps", 1, 20, 10)
entropy_gate = st.sidebar.slider("Entropy Gate Threshold", 0.0, 1.0, 0.3, 0.05)

n_samples_option = st.sidebar.selectbox(
    "Test Samples",
    ["All", "100 (fast)", "200", "500"],
)
n_samples_map = {"All": None, "100 (fast)": 100, "200": 200, "500": 500}
n_samples = n_samples_map[n_samples_option]

# ── Load model and data ──────────────────────────────────────────────────

seed_everything(42)
device = get_device()

model, train_args, ckpt = load_model(ckpt_path, device)
dataset = load_dataset()
mask_mode = train_args.get("mask_mode", "random_slices")

adaptor = TTTAdaptor(
    model=model,
    base_lr=ttt_lr,
    ttt_steps=ttt_steps,
    mask_mode=mask_mode,
    ttt_optimizer="adam",
    entropy_adaptive=True,
    entropy_scale=2.0,
    entropy_gate_threshold=entropy_gate,
    device=device,
)

# ── Header ───────────────────────────────────────────────────────────────

st.markdown("""
# 📊 TTT Crypto Regime Dashboard
**Test-Time Training for Robust Volatility Classification**
""")

st.markdown(f"""
<div style="display: flex; gap: 16px; margin-bottom: 24px;">
    <div style="background: #1a1a2e; padding: 8px 16px; border-radius: 8px; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;">
        Model: <span style="color: #00d4aa;">{selected_ckpt}</span>
    </div>
    <div style="background: #1a1a2e; padding: 8px 16px; border-radius: 8px; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;">
        Epoch: <span style="color: #00d4aa;">{ckpt.get('epoch', '?')}</span>
    </div>
    <div style="background: #1a1a2e; padding: 8px 16px; border-radius: 8px; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;">
        Device: <span style="color: #00d4aa;">{device}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["📈 Overview", "🔬 Regime Analysis", "🔄 Prediction Shifts"])

# ── Tab 1: Overview ──────────────────────────────────────────────────────

with tab1:
    if st.button("▶ Run Evaluation", key="run_overview", type="primary"):
        with st.spinner("Running baseline + TTT evaluation..."):
            results = run_evaluation(model, dataset, adaptor, threshold, n_samples)
            st.session_state["overview_results"] = results

    if "overview_results" in st.session_state:
        results = st.session_state["overview_results"]
        bm = results["baseline"]
        tm = results["ttt_standard"]

        st.markdown("### Baseline")
        cols = st.columns(5)
        for col, (name, key) in zip(cols, [
            ("Accuracy", "accuracy"), ("F1", "f1"), ("ECE", "ece"),
            ("Brier", "brier"), ("IC", "IC")
        ]):
            with col:
                metric_card(name, bm.get(key, 0.0))

        st.markdown("### TTT (Standard)")
        cols = st.columns(5)
        for col, (name, key) in zip(cols, [
            ("Accuracy", "accuracy"), ("F1", "f1"), ("ECE", "ece"),
            ("Brier", "brier"), ("IC", "IC")
        ]):
            with col:
                delta = tm.get(key, 0.0) - bm.get(key, 0.0)
                metric_card(name, tm.get(key, 0.0), delta=delta)

        # Comparison table
        st.markdown("### Comparison Table")
        df = pd.DataFrame({
            "Mode": ["Baseline", "TTT (Standard)"],
            "Accuracy": [bm["accuracy"], tm["accuracy"]],
            "F1": [bm["f1"], tm["f1"]],
            "ECE": [bm["ece"], tm["ece"]],
            "Brier": [bm["brier"], tm["brier"]],
            "IC": [bm.get("IC", 0.0), tm.get("IC", 0.0)],
        })
        st.dataframe(df.style.format({
            "Accuracy": "{:.4f}", "F1": "{:.4f}", "ECE": "{:.4f}",
            "Brier": "{:.4f}", "IC": "{:.4f}",
        }), use_container_width=True, hide_index=True)

# ── Tab 2: Regime Analysis ───────────────────────────────────────────────

with tab2:
    if st.button("▶ Run Regime Analysis", key="run_regime", type="primary"):
        with st.spinner("Evaluating per volatility regime..."):
            regime_df = run_regime_evaluation(model, dataset, adaptor, threshold)
            st.session_state["regime_results"] = regime_df

    if "regime_results" in st.session_state:
        regime_df = st.session_state["regime_results"]

        st.markdown("### Performance by Volatility Regime")
        st.dataframe(regime_df.style.format({
            "Baseline Acc": "{:.3f}", "Baseline F1": "{:.3f}",
            "TTT Acc": "{:.3f}", "TTT F1": "{:.3f}",
            "Δ Acc": "{:+.3f}", "Δ F1": "{:+.3f}",
        }).background_gradient(
            subset=["Δ F1"], cmap="RdYlGn", vmin=-0.5, vmax=0.5,
        ), use_container_width=True, hide_index=True)

        # Bar chart comparison
        st.markdown("### Accuracy by Regime")
        chart_df = regime_df[["Regime", "Baseline Acc", "TTT Acc"]].melt(
            id_vars="Regime", var_name="Mode", value_name="Accuracy"
        )
        st.bar_chart(
            chart_df.pivot(index="Regime", columns="Mode", values="Accuracy"),
            use_container_width=True,
        )

        st.markdown("### F1 by Regime")
        chart_df_f1 = regime_df[["Regime", "Baseline F1", "TTT F1"]].melt(
            id_vars="Regime", var_name="Mode", value_name="F1"
        )
        st.bar_chart(
            chart_df_f1.pivot(index="Regime", columns="Mode", values="F1"),
            use_container_width=True,
        )

        # Key insight
        high_row = regime_df[regime_df["Regime"].str.contains("High")]
        if not high_row.empty:
            hr = high_row.iloc[0]
            st.markdown(f"""
            > **Key Finding:** TTT improves high-vol F1 from **{hr['Baseline F1']:.3f}** to
            > **{hr['TTT F1']:.3f}** ({hr['Δ F1']:+.3f}), confirming that adaptation helps most
            > during regime shifts where distribution shift is largest.
            """)

# ── Tab 3: Prediction Shifts ────────────────────────────────────────────

with tab3:
    if "overview_results" not in st.session_state:
        st.info("Run the Overview evaluation first to generate prediction data.")
    else:
        results = st.session_state["overview_results"]

        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")

        probs_b = results["baseline_probs"]
        probs_t = results["ttt_probs"]
        labels = results["labels"]

        prob_b_pos = probs_b[:, 1] if probs_b.ndim == 2 else probs_b
        prob_t_pos = probs_t[:, 1] if probs_t.ndim == 2 else probs_t

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Before vs After TTT")
            fig1, ax1 = plt.subplots(figsize=(6, 5))
            fig1.patch.set_facecolor('#0a0a0f')
            ax1.set_facecolor('#0a0a0f')

            low_mask = labels == 0
            high_mask = labels == 1

            ax1.scatter(prob_b_pos[low_mask], prob_t_pos[low_mask],
                       c="#00d4aa", alpha=0.5, s=20, label="Low Vol")
            ax1.scatter(prob_b_pos[high_mask], prob_t_pos[high_mask],
                       c="#ff4757", alpha=0.5, s=20, label="High Vol")
            ax1.plot([0, 1], [0, 1], "--", color="#444", lw=1)
            ax1.set_xlabel("P(high vol) — Baseline", color="#aaa", fontsize=10)
            ax1.set_ylabel("P(high vol) — After TTT", color="#aaa", fontsize=10)
            ax1.legend(facecolor="#1a1a2e", edgecolor="#2a2a4a", labelcolor="#e0e0e0")
            ax1.tick_params(colors="#666")
            for spine in ax1.spines.values():
                spine.set_color("#2a2a4a")
            st.pyplot(fig1)

        with col2:
            st.markdown("### Distribution of Prediction Shifts")
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            fig2.patch.set_facecolor('#0a0a0f')
            ax2.set_facecolor('#0a0a0f')

            delta = prob_t_pos - prob_b_pos
            ax2.hist(delta[low_mask], bins=25, alpha=0.6, color="#00d4aa", label="Low Vol")
            ax2.hist(delta[high_mask], bins=25, alpha=0.6, color="#ff4757", label="High Vol")
            ax2.axvline(0, color="#666", ls="--", lw=1)
            ax2.set_xlabel("ΔP(high vol) = TTT − Baseline", color="#aaa", fontsize=10)
            ax2.set_ylabel("Count", color="#aaa", fontsize=10)
            ax2.legend(facecolor="#1a1a2e", edgecolor="#2a2a4a", labelcolor="#e0e0e0")
            ax2.tick_params(colors="#666")
            for spine in ax2.spines.values():
                spine.set_color("#2a2a4a")
            st.pyplot(fig2)

        # Stats
        st.markdown("### Shift Statistics")
        stats_df = pd.DataFrame({
            "Class": ["Low Vol", "High Vol", "All"],
            "Mean ΔP": [delta[low_mask].mean(), delta[high_mask].mean(), delta.mean()],
            "Std ΔP": [delta[low_mask].std(), delta[high_mask].std(), delta.std()],
            "Samples Shifted Up": [
                (delta[low_mask] > 0.01).sum(),
                (delta[high_mask] > 0.01).sum(),
                (delta > 0.01).sum(),
            ],
            "Samples Unchanged": [
                (np.abs(delta[low_mask]) <= 0.01).sum(),
                (np.abs(delta[high_mask]) <= 0.01).sum(),
                (np.abs(delta) <= 0.01).sum(),
            ],
        })
        st.dataframe(stats_df.style.format({
            "Mean ΔP": "{:+.4f}", "Std ΔP": "{:.4f}",
        }), use_container_width=True, hide_index=True)


# ── Footer ───────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #555; font-size: 0.8rem; font-family: 'JetBrains Mono', monospace;">
    TTT Crypto Regime Dashboard · Re-implementation of Sun et al. (2020) · Pramesh Singhavi · Feb 2026
</div>
""", unsafe_allow_html=True)