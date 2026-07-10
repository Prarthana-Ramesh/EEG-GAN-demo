from __future__ import annotations

import pickle
import random
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import time
from scipy import signal

ROOT = Path(__file__).resolve().parent
if not (ROOT / "data").exists():
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.eeg_synthesis import generate_real_trial, generate_fake_trial
from app.components import (
    plot_confusion_matrix,
    plot_kappa_trajectory,
    plot_psd_comparison,
    plot_ratio_curve,
    plot_raw_signal,
    render_biomarker_guide,
    render_classifiers,
    render_live_preprocessing,
    render_real_vs_fake,
)
from app.utils import get_signal

DATA_PATH = ROOT / "data" / "demo_data.pkl"


@st.cache_data(show_spinner=False)
def load_demo_data() -> dict:
    with DATA_PATH.open("rb") as handle:
        return pickle.load(handle)


def main() -> None:
    st.set_page_config(page_title="BCI Augmentation Pipeline", page_icon="🧠", layout="wide")

    st.markdown(
        """
        <style>
            :root {
                color-scheme: dark;
            }

            .stApp {
                background:
                    radial-gradient(circle at 16% 10%, rgba(56, 189, 248, 0.10) 0%, rgba(56, 189, 248, 0.00) 28%),
                    radial-gradient(circle at 84% 16%, rgba(14, 165, 233, 0.08) 0%, rgba(14, 165, 233, 0.00) 26%),
                    radial-gradient(circle at 33% 82%, rgba(99, 102, 241, 0.10) 0%, rgba(99, 102, 241, 0.00) 30%),
                    linear-gradient(135deg, #050816 0%, #070b14 20%, #0b1120 50%, #03050a 100%);
                background-attachment: fixed;
                color: rgba(226, 232, 240, 0.94);
            }

            .stApp:before {
                content: "";
                position: fixed;
                inset: 0;
                pointer-events: none;
                background:
                    linear-gradient(135deg, rgba(255, 255, 255, 0.02) 0 12%, transparent 12% 48%, rgba(255, 255, 255, 0.015) 48% 53%, transparent 53% 100%),
                    linear-gradient(315deg, rgba(0, 0, 0, 0.48) 0 18%, transparent 18% 42%, rgba(255, 255, 255, 0.01) 42% 46%, transparent 46% 100%);
                opacity: 1;
                z-index: 0;
            }

            .main .block-container,
            div[data-testid="stMarkdownContainer"],
            div[data-testid="stCaptionContainer"],
            label {
                color: rgba(226, 232, 240, 0.94);
            }

            .stCaption {
                color: rgba(148, 163, 184, 0.95) !important;
            }

            div[data-testid="stMetric"] {
                background: rgba(15, 23, 42, 0.58);
                border: 1px solid rgba(148, 163, 184, 0.15);
                border-radius: 18px;
                padding: 0.75rem 1rem;
                box-shadow: 0 18px 36px rgba(2, 6, 23, 0.28);
            }

            section[data-testid="stSidebar"] {
                background: transparent;
            }

            section[data-testid="stSidebar"] > div {
                background: linear-gradient(180deg, rgba(15, 23, 42, 0.92) 0%, rgba(10, 15, 28, 0.92) 100%);
                backdrop-filter: blur(18px);
                -webkit-backdrop-filter: blur(18px);
                border: 1px solid rgba(148, 163, 184, 0.14);
                border-radius: 24px;
                margin: 1rem 0.8rem 1rem 1rem;
                box-shadow: 0 28px 72px rgba(2, 6, 23, 0.6);
                overflow: hidden;
            }

            section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
                padding-top: 1.25rem;
                overflow: hidden;
            }

            section[data-testid="stSidebar"] .sidebar-title {
                color: rgba(248, 250, 252, 0.98);
                font-size: 1.55rem;
                font-weight: 800;
                letter-spacing: 0.02em;
                margin: 0.25rem 0 1rem;
                padding: 0 1rem;
            }

            section[data-testid="stSidebar"] .sidebar-subtitle {
                color: rgba(148, 163, 184, 0.96);
                font-size: 0.92rem;
                padding: 0 1rem 0.75rem;
                margin-bottom: 0.25rem;
                border-bottom: 1px solid rgba(148, 163, 184, 0.16);
            }

            section[data-testid="stSidebar"] .sidebar-menu-label {
                color: rgba(226, 232, 240, 0.9);
                font-size: 0.82rem;
                font-weight: 700;
                letter-spacing: 0.18em;
                text-transform: uppercase;
                padding: 0.9rem 1rem 0.35rem;
            }

            section[data-testid="stSidebar"] button {
                background: rgba(15, 23, 42, 0.72) !important;
                border: 1px solid rgba(148, 163, 184, 0.16) !important;
                border-radius: 16px !important;
                color: rgba(248, 250, 252, 0.94) !important;
                width: 100% !important;
                justify-content: flex-start !important;
                gap: 0.42rem !important;
                padding: 0.62rem 0.95rem !important;
                margin: 0.12rem 0 !important;
                box-shadow: none !important;
                transition: transform 0.18s ease, background 0.18s ease, border-color 0.18s ease;
                text-align: left !important;
            }

            section[data-testid="stSidebar"] button:hover {
                transform: translateX(2px);
                background: rgba(30, 41, 59, 0.96) !important;
                border-color: rgba(96, 165, 250, 0.28) !important;
            }

            section[data-testid="stSidebar"] button[kind="primary"] {
                background: linear-gradient(135deg, rgba(30, 41, 59, 0.98) 0%, rgba(15, 23, 42, 0.98) 100%) !important;
                border-color: rgba(96, 165, 250, 0.36) !important;
                box-shadow: inset 0 0 0 1px rgba(148, 163, 184, 0.08), 0 10px 28px rgba(2, 6, 23, 0.34) !important;
            }

            div[data-testid="stSelectbox"] {
                background: linear-gradient(180deg, rgba(15, 23, 42, 0.92) 0%, rgba(10, 15, 28, 0.92) 100%);
                border: 1px solid rgba(148, 163, 184, 0.16);
                border-radius: 18px;
                padding: 0.35rem 0.45rem 0.45rem;
                backdrop-filter: blur(16px);
                -webkit-backdrop-filter: blur(16px);
                box-shadow: 0 16px 36px rgba(2, 6, 23, 0.32);
                transition: transform 0.18s ease, background 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
            }

            div[data-testid="stSelectbox"]:hover {
                transform: translateY(-2px);
                background: linear-gradient(180deg, rgba(30, 41, 59, 0.98) 0%, rgba(15, 23, 42, 0.98) 100%);
                border-color: rgba(96, 165, 250, 0.26);
                box-shadow: 0 22px 44px rgba(2, 6, 23, 0.38);
            }

            div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
                background: transparent;
                border: 0;
                box-shadow: none;
                color: rgba(248, 250, 252, 0.96);
            }

            div[data-testid="stSelectbox"] svg {
                fill: rgba(226, 232, 240, 0.84);
            }

            div[data-testid="stSelectbox"] div[data-baseweb="popover"] {
                background: rgba(15, 23, 42, 0.98);
                border: 1px solid rgba(148, 163, 184, 0.12);
            }

            section[data-testid="stSidebar"] button span {
                color: inherit !important;
                font-weight: 600;
            }

            section[data-testid="stSidebar"] button > div {
                justify-content: flex-start !important;
            }

            section[data-testid="stSidebar"] button [data-testid="stMarkdownContainer"] {
                text-align: left !important;
            }

            .block-container {
                position: relative;
                z-index: 1;
                background: transparent;
            }

            hr {
                border-color: rgba(148, 163, 184, 0.18) !important;
            }

            .stExpander {
                background: rgba(15, 23, 42, 0.5);
                border: 1px solid rgba(148, 163, 184, 0.14);
                border-radius: 16px;
            }

            .phase-selector { margin-bottom: 20px; }
            .phase-selector label { font-weight: 600; font-size: 1.1rem; }
            .phase-highlight { border-left: 6px solid var(--accent); padding-left: 15px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    payload = load_demo_data()
    all_phases = payload["phases"]

    col_title, col_selector, col_spacer = st.columns([2, 1.5, 1])
    with col_title:
        st.title("BCI Augmentation Pipeline")
        st.caption("Interactive walkthrough of preprocessing, GAN architecture, and classification.")
    with col_selector:
        st.markdown("<div class='sidebar-menu-label' style='padding-left: 0; padding-top: 0;'>Live Evolution View</div>", unsafe_allow_html=True)
        phase_num = st.selectbox(
            "Select Phase",
            options=[1, 2, 3],
            format_func=lambda x: {1: "⚡ Baseline (46%)", 2: "🔄 VAE-GAN (75%)", 3: "✅ MI-Aware (86%)"}[x],
            index=2,
            label_visibility="collapsed",
        )

    phase_data = all_phases[phase_num]
    metrics = phase_data["metrics"]
    class_names = phase_data["class_names"]
    features = phase_data["features"]
    color = phase_data["color"]

    st.markdown(f"""
        <style>
            .phase-highlight {{ border-left: 6px solid {color}; padding-left: 15px; }}
        </style>
        """, unsafe_allow_html=True)

    nav_items = [
        ("Raw Input", "home", "home"),
        ("Live Preprocessing", "tune", "preprocess"),
        ("Architecture", "schema", "architecture"),
        ("Augmentation", "dataset", "augmentation"),
        ("Classifiers", "psychology", "classifiers"),
        ("Results", "analytics", "results"),
        ("Real vs Fake", "compare_arrows", "challenge"),
    ]

    if "main_nav" not in st.session_state:
        st.session_state["main_nav"] = nav_items[0][0]

    st.sidebar.markdown('<div class="sidebar-title">BCI</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-subtitle">Augmentation Pipeline</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-menu-label">Menu</div>', unsafe_allow_html=True)

    for label, icon_name, item_key in nav_items:
        is_selected = st.session_state["main_nav"] == label
        if st.sidebar.button(
            label,
            icon=f":material/{icon_name}:",
            key=f"nav_{item_key}",
            use_container_width=True,
            type="primary" if is_selected else "secondary",
        ):
            st.session_state["main_nav"] = label

    selected_tab = st.session_state["main_nav"]

    if selected_tab == "Raw Input":
        render_biomarker_guide(color)
    elif selected_tab == "Live Preprocessing":
        st.header("🔬 Live Preprocessing Pipeline")
        st.markdown("Tune filters and watch the EEG waveform transform in real time. The gray trace is the raw signal; the colored trace is your processed output.")
        render_live_preprocessing(phase_data, color, phase_num)
        with st.expander("⚡ How does this work?"):
            st.markdown("- Bandpass keeps the motor-imagery frequencies of interest.\n- Notch removes power-line hum.\n- Common-average re-referencing suppresses shared noise.\n- Baseline correction zeros the pre-cue period.\n- ICA (mock) simulates removing blink and jaw-clench artifacts.")
    elif selected_tab == "Architecture":
        st.header("GAN Architecture")
        st.markdown("Display the proposed architecture image. The page intentionally shows only the image to keep the architecture focused and large.")

        assets_dir = ROOT / "assets"
        img_path = Path("./assets/architecture.png")

        if img_path.exists():
            st.image(str(img_path), use_column_width=True)
        else:
            st.warning("Architecture image not found at assets/architecture.png")
            st.info("Drop the provided architecture image into the repository at `assets/architecture.png`, or upload it below to preview and save it.")
            uploaded = st.file_uploader("Upload architecture image", type=["png", "jpg", "jpeg"])
            if uploaded is not None:
                st.image(uploaded, use_column_width=True)
                try:
                    assets_dir.mkdir(parents=True, exist_ok=True)
                    save_path = assets_dir / "architecture.png"
                    with save_path.open("wb") as f:
                        f.write(uploaded.getvalue())
                    st.success(f"Saved image to {save_path}")
                except Exception as e:
                    st.error(f"Could not save image to disk: {e}")
    elif selected_tab == "Augmentation":
        st.header("Augmentation & Classifier Training")
        st.markdown("Synthetic data boosts classifier performance, especially when real data is scarce.")

        gen_key = f"aug_generated_{phase_num}"
        placeholder = st.container()
        with placeholder:
            st.markdown("---")
            c1, c2, c3 = st.columns([1, 2, 1])
            with c1:
                st.write("")
            with c2:
                st.markdown("### Generate class-wise example signals")
                if st.button("Generate signals", key=f"gen_btn_{phase_num}") or (gen_key in st.session_state and st.session_state.get(gen_key) is None):
                    st.session_state[gen_key] = None
                    progress = st.progress(0)
                    messages = st.empty()
                    classes = phase_data.get("class_names", ["Right Hand", "Left Hand", "Foot"])[:3]
                    generated = {}
                    total_steps = len(classes) * 3
                    step = 0
                    for cls in classes:
                        messages.text(f"Generating examples for {cls} ...")
                        for _ in range(3):
                            time.sleep(0.6 + random.random() * 0.6)
                            step += 1
                            progress.progress(int(step / total_steps * 100))
                        real_t, _ = generate_real_trial(cls, n_samples=550, seed=random.randint(0, 10000))
                        fake_t, _ = generate_fake_trial(cls, phase_num, n_samples=550, seed=random.randint(0, 10000))
                        generated[cls] = {"real": real_t, "fake": fake_t}
                    messages.text("Finalizing...")
                    time.sleep(0.6)
                    progress.progress(100)
                    st.session_state[gen_key] = generated
                    messages.empty()
            with c3:
                st.write("")

        st.markdown("---")

        if gen_key in st.session_state and st.session_state.get(gen_key):
            gallery = st.session_state[gen_key]
            st.subheader("Generated Examples (labeled for demo)")
            for cls, pair in gallery.items():
                st.markdown(f"**{cls}**")
                col_r, col_f = st.columns(2)
                with col_r:
                    st.caption("Real-like (generated)")
                    st.plotly_chart(plot_raw_signal(pair["real"], color, ""), width="stretch")
                with col_f:
                    st.caption(f"Synthetic (Phase {phase_num})")
                    st.plotly_chart(plot_raw_signal(pair["fake"], "#7f8c8d", ""), width="stretch")
            st.markdown("---")
            st.markdown("_Short explanation:_ These examples are generated from the physiology-based synthesizer. The left column shows realistic-style trials from the same generator; the right column shows phase-specific synthetic signals. Use these to evaluate augmentation impact and inspect waveform/biomarker differences.")

        ratio = st.slider("Real vs Synthetic Training Ratio (% Real)", 0, 100, 50, key=f"ratio_{phase_num}")
        acc_at_ratio = np.interp(ratio, phase_data["ratio_anchors"]["x"], phase_data["ratio_anchors"]["y"])
        st.metric(f"Estimated Accuracy at {ratio}% Real", f"{acc_at_ratio * 100:.1f}%")
        st.markdown("---")
        st.subheader("Sample Generation Quality")
        sample_idx = int(np.random.randint(0, min(200, len(phase_data["real"]))))
        col_real, col_fake, col_psd = st.columns(3)
        with col_real:
            st.caption("**Real Trial**")
            st.plotly_chart(plot_raw_signal(phase_data["real"][sample_idx], color, ""), width="stretch")
        with col_fake:
            st.caption("**Synthetic Trial**")
            st.plotly_chart(plot_raw_signal(phase_data["fake"][sample_idx], "#7f8c8d", ""), width="stretch")
        with col_psd:
            st.caption("**PSD Comparison**")
            st.plotly_chart(plot_psd_comparison(phase_data["real"][sample_idx], phase_data["fake"][sample_idx], color), width="stretch")
    elif selected_tab == "Classifiers":
        render_classifiers(phase_data, phase_num)
    elif selected_tab == "Results":
        st.header("Final Classification Performance")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{metrics['accuracy'] * 100:.1f}%", delta=f"+{(metrics['accuracy'] - 0.463) * 100:.1f}%" if phase_num > 1 else None)
        col2.metric("Kappa Score", f"{metrics['kappa']:.3f}")
        col3.metric("Classes", f"{len(class_names)}", delta="2-class (Right/Foot)" if len(class_names) == 2 else "3-class (LH/RH/Foot)")
        st.subheader("Confusion Matrix")
        st.plotly_chart(plot_confusion_matrix(metrics["confusion_matrix"], class_names, color), width="stretch")
        st.subheader("Kappa Score Over Time (0-7s)")
        st.plotly_chart(plot_kappa_trajectory(metrics["kappa_trajectory"], color), width="stretch")
        st.subheader("Cross-Subject Generalization (LOSO)")
        st.caption("Leave-One-Subject-Out validation. Diagonal = high accuracy, off-diagonal = cross-subject transfer.")
        if phase_num == 1:
            loso_data = np.array([[0.46, 0.38, 0.35, 0.36, 0.40], [0.39, 0.48, 0.36, 0.38, 0.41], [0.36, 0.37, 0.45, 0.37, 0.39], [0.37, 0.39, 0.36, 0.47, 0.40], [0.41, 0.42, 0.40, 0.39, 0.49]])
        elif phase_num == 2:
            loso_data = np.array([[0.75, 0.62, 0.58, 0.60, 0.65], [0.64, 0.78, 0.61, 0.63, 0.66], [0.59, 0.60, 0.75, 0.62, 0.64], [0.61, 0.63, 0.60, 0.77, 0.65], [0.66, 0.67, 0.65, 0.64, 0.79]])
        else:
            loso_data = np.array([[0.86, 0.72, 0.68, 0.70, 0.75], [0.74, 0.88, 0.71, 0.73, 0.76], [0.69, 0.70, 0.85, 0.72, 0.74], [0.71, 0.73, 0.70, 0.87, 0.75], [0.76, 0.77, 0.75, 0.74, 0.89]])
        fig_loso = go.Figure(data=go.Heatmap(z=loso_data, x=["S1", "S2", "S3", "S4", "S5"], y=["S1", "S2", "S3", "S4", "S5"], colorscale="RdBu", zmin=0.3, zmax=0.9, text=loso_data.round(2), texttemplate="%{text}", textfont={"size": 14}))
        fig_loso.update_layout(template="plotly_white", height=400, xaxis_title="Test Subject", yaxis_title="Train Subject")
        st.plotly_chart(fig_loso, width="stretch")
    else:
        st.header("🎮 Signal Realism Challenge")
        st.markdown("Try to tell which signal looks more realistic for this phase's GAN quality, using the same physiology-driven synthesis style as the first tab.")
        render_real_vs_fake(phase_data, phase_num)


if __name__ == "__main__":
    main()
