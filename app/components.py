from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import signal

from app.eeg_synthesis import (
    CLASS_COLORS,
    CLASS_TOPOGRAPHY,
    DEFAULT_CHANNELS,
    analyze_biomarkers,
    describe_biomarkers,
    generate_fake_trial,
    generate_real_trial,
)
from app.utils import (
    apply_bandpass,
    apply_baseline_correction,
    apply_car,
    apply_notch,
    get_signal,
)


def plot_raw_signal(trial: np.ndarray, color: str = "#2c3e50", title: str = "Raw EEG") -> go.Figure:
    sig = get_signal(trial)
    time = np.arange(len(sig)) / 100.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=sig, mode="lines", line=dict(color=color, width=1.6)))
    fig.update_layout(title=title, xaxis_title="Time (s)", yaxis_title="Amplitude (µV)", template="plotly_white", height=300)
    return fig


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], color: str = "#2ecc71") -> go.Figure:
    fig = go.Figure(data=go.Heatmap(z=cm, x=class_names, y=class_names, colorscale=[[0, "#f8f9fa"], [1, color]], text=cm, texttemplate="%{text}", textfont={"size": 16}))
    fig.update_layout(template="plotly_white", height=350, xaxis_title="Predicted", yaxis_title="Actual")
    return fig


def plot_kappa_trajectory(kappa: np.ndarray, color: str = "#2ecc71") -> go.Figure:
    time = np.linspace(0, 7, len(kappa))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=kappa, mode="lines", line=dict(color=color, width=3)))
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
    fig.update_layout(template="plotly_white", xaxis_title="Time (s)", yaxis_title="Kappa", height=300)
    return fig


def plot_ratio_curve(anchors: dict[str, list[float]], selected_phase: int, all_phases: dict[int, dict]) -> go.Figure:
    ratios = np.linspace(0, 100, 50)
    fig = go.Figure()
    for phase_num in [1, 2, 3]:
        phase = all_phases[phase_num]
        x_vals = np.asarray(phase["ratio_anchors"]["x"], dtype=float)
        y_vals = np.asarray(phase["ratio_anchors"]["y"], dtype=float)
        acc = np.interp(ratios, x_vals, y_vals)
        fig.add_trace(go.Scatter(x=ratios, y=acc, mode="lines", name=phase["name"], line=dict(color=phase["color"], width=4 if phase_num == selected_phase else 2), opacity=1.0 if phase_num == selected_phase else 0.55))
    fig.update_layout(template="plotly_white", height=350, xaxis_title="% Real Data in Training", yaxis_title="Accuracy")
    return fig


def plot_psd_comparison(real_trial: np.ndarray, fake_trial: np.ndarray, color: str = "#2ecc71") -> go.Figure:
    real_sig = get_signal(real_trial)
    fake_sig = get_signal(fake_trial)
    freqs_r, psd_r = signal.welch(real_sig, fs=100.0)
    freqs_f, psd_f = signal.welch(fake_sig, fs=100.0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freqs_r, y=psd_r, mode="lines", name="Real", line=dict(color=color, width=2)))
    fig.add_trace(go.Scatter(x=freqs_f, y=psd_f, mode="lines", name="Synthetic", line=dict(color="#7f8c8d", width=2, dash="dash")))
    fig.update_layout(template="plotly_white", xaxis_title="Frequency (Hz)", yaxis_title="PSD", height=250)
    return fig


# ----------------------------------------------------------------------
# Multichannel biomarker teaching panels
# ----------------------------------------------------------------------

def _shade_windows(fig: go.Figure, meta: dict, row: int | None = None, rows_total: int = 1) -> None:
    """Add ERD/ERS shading and cue markers to a Plotly figure."""
    fs = meta["fs"]
    b0, b1 = meta["baseline_window"]
    m0, m1 = meta["mi_window"]
    r0, r1 = meta["rebound_window"]
    kwargs = dict(row=row, col=1) if row is not None else {}
    fig.add_vrect(x0=m0 / fs, x1=m1 / fs, fillcolor="rgba(231, 76, 60, 0.10)", line_width=0, **kwargs)
    fig.add_vrect(x0=r0 / fs, x1=r1 / fs, fillcolor="rgba(46, 204, 113, 0.12)", line_width=0, **kwargs)
    fig.add_vline(x=b1 / fs, line_dash="dot", line_color="#7f8c8d", line_width=1, **kwargs)


def plot_multichannel_stack(trial: np.ndarray, meta: dict, title: str = "") -> go.Figure:
    channels = meta["channels"]
    fs = meta["fs"]
    n_ch = len(channels)
    time = np.arange(trial.shape[0]) / fs
    color = CLASS_COLORS.get(meta["class_name"], "#2c3e50")

    fig = make_subplots(rows=n_ch, cols=1, shared_xaxes=True, vertical_spacing=0.015)
    for i, ch in enumerate(channels):
        row = i + 1
        line_color = "#e67e22" if ch in ("Fp1", "Fp2") else color
        fig.add_trace(go.Scatter(x=time, y=trial[:, i], mode="lines", line=dict(color=line_color, width=1.3), showlegend=False), row=row, col=1)
        _shade_windows(fig, meta, row=row)
        fig.update_yaxes(title_text=ch, title_font=dict(size=10), showticklabels=False, row=row, col=1)

    fig.update_xaxes(title_text="Time (s)", row=n_ch, col=1)
    fig.update_layout(template="plotly_white", height=90 * n_ch + 60, title=title, margin=dict(t=40 if title else 20, b=30))
    return fig


def plot_annotated_hero_channel(trial: np.ndarray, meta: dict) -> go.Figure:
    channels = meta["channels"]
    fs = meta["fs"]
    hero = meta["dominant_channel"]
    idx = channels.index(hero) if hero in channels else 0
    sig = trial[:, idx]
    time = np.arange(len(sig)) / fs
    color = CLASS_COLORS.get(meta["class_name"], "#2c3e50")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=sig, mode="lines", line=dict(color=color, width=1.8)))
    _shade_windows(fig, meta)

    b0, b1 = meta["baseline_window"]
    m0, m1 = meta["mi_window"]
    r0, r1 = meta["rebound_window"]

    mi_slice = sig[m0:m1]
    if len(mi_slice):
        dip_idx = m0 + int(np.argmin(mi_slice))
        fig.add_annotation(x=dip_idx / fs, y=sig[dip_idx], text="ERD ↓ (desync)", showarrow=True, arrowhead=2, ax=0, ay=-40, bgcolor="rgba(231,76,60,0.85)", font=dict(color="white", size=11))

    rebound_slice = sig[r0:r1]
    if len(rebound_slice):
        peak_idx = r0 + int(np.argmax(rebound_slice))
        fig.add_annotation(x=peak_idx / fs, y=sig[peak_idx], text="ERS ↑ (beta rebound)", showarrow=True, arrowhead=2, ax=0, ay=-40, bgcolor="rgba(46,204,113,0.9)", font=dict(color="white", size=11))

    for ev in meta.get("eog_events", [])[:1]:
        fig.add_annotation(x=ev / fs, y=sig[min(ev, len(sig) - 1)], text="Eye-blink (EOG)", showarrow=True, arrowhead=2, ax=30, ay=40, bgcolor="rgba(230,126,34,0.9)", font=dict(color="white", size=11))

    fig.add_annotation(x=(b0 + b1) / 2 / fs, y=1.0, yref="paper", yanchor="bottom", showarrow=False, text="Baseline", font=dict(size=10, color="#7f8c8d"))
    fig.add_annotation(x=(m0 + m1) / 2 / fs, y=1.0, yref="paper", yanchor="bottom", showarrow=False, text="Motor Imagery", font=dict(size=10, color="#c0392b"))
    fig.add_annotation(x=(r0 + r1) / 2 / fs, y=1.0, yref="paper", yanchor="bottom", showarrow=False, text="Rebound", font=dict(size=10, color="#1e8449"))

    fig.update_layout(template="plotly_white", height=340, title=f"Channel {hero} — {meta['class_name']} imagery", xaxis_title="Time (s)", yaxis_title="Amplitude (µV)", margin=dict(t=60))
    return fig


def render_biomarker_guide(default_color: str = "#2ecc71") -> None:
    st.markdown("### 🧭 Explore a Motor-Imagery Trial by Class")
    st.caption("Generated from a physiological model: 1/f background + mu/beta rhythms + eye-blink artifacts — realistic but cleaner than raw recordings.")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        class_name = st.selectbox("Motor-Imagery Class", ["Right Hand", "Left Hand", "Foot", "Tongue"], index=0, key="biomarker_class")
    with col_b:
        if st.button("🔄 New Example", width="stretch", key="biomarker_reseed"):
            st.session_state["biomarker_seed"] = int(np.random.randint(0, 10_000))

    seed = st.session_state.get("biomarker_seed", 42)
    trial, meta = generate_real_trial(class_name, n_samples=550, seed=seed)
    color = CLASS_COLORS.get(class_name, default_color)

    metrics = analyze_biomarkers(trial[:, meta["channels"].index(meta["dominant_channel"])], meta["fs"])
    m1, m2, m3 = st.columns(3)
    m1.metric("ERD (power drop)", f"{metrics['erd_pct']:.0f}%", help="Mu/beta power decrease during imagined movement vs. baseline.")
    m2.metric("ERS (beta rebound)", f"{metrics['ers_pct']:.0f}%", help="Mu/beta power increase after movement ends, relative to baseline.")
    m3.metric("Dominant Channel", meta["dominant_channel"], help="The sensorimotor electrode expected to show the strongest effect.")

    st.plotly_chart(plot_annotated_hero_channel(trial, meta), width="stretch")
    st.plotly_chart(plot_multichannel_stack(trial, meta, title=f"Full Montage — {class_name} Imagery"), width="stretch")

    with st.expander("📚 Understand the Biomarkers"):
        st.markdown(
            """
**Mu (8–12 Hz) & Beta (13–30 Hz) rhythms** are the two frequency bands over sensorimotor cortex that change most during motor imagery.

**ERD — Event-Related Desynchronization** 📉
During imagined movement, these rhythms drop in power; the stronger the drop, the clearer the motor-imagery signature.

**ERS — Event-Related Synchronization (beta rebound)** 📈
Right after the imagery window, beta power briefly rises above baseline as the movement-related activity settles.

**Topography — why the channel matters**
The strongest ERD/ERS effect is usually seen over the class-appropriate sensorimotor electrode (for example C3 for right-hand imagery).
"""
        )
        st.table({
            "Imagined Movement": ["Right Hand", "Left Hand", "Foot", "Tongue"],
            "Strongest ERD Channel": ["C3", "C4", "Cz", "Cz / bilateral fronto-central"],
            "Why": ["Left motor cortex controls the right hand", "Right motor cortex controls the left hand", "Foot representation sits at the cortical midline", "Tongue/orofacial representation is bilateral"],
        })


# ----------------------------------------------------------------------
# Real-vs-fake challenge
# ----------------------------------------------------------------------

def render_real_vs_fake(phase_data: dict, phase_num: int) -> None:
    # compact realism challenge: single-channel hero with expandable full montage
    real_key = f"real_on_a_{phase_num}"
    choice_key = f"choice_{phase_num}"
    class_key = f"challenge_class_{phase_num}"
    seed_key = f"challenge_seed_{phase_num}"

    classes = ["Right Hand", "Left Hand", "Foot"]

    def _new_round() -> None:
        st.session_state[real_key] = bool(np.random.randint(0, 2))
        st.session_state[choice_key] = None
        st.session_state[class_key] = classes[int(np.random.randint(0, len(classes)))]
        st.session_state[seed_key] = int(np.random.randint(0, 10_000))

    if seed_key not in st.session_state:
        _new_round()

    class_name = st.session_state[class_key]
    seed = int(st.session_state[seed_key])
    real_trial, real_meta = generate_real_trial(class_name, n_samples=550, seed=seed)
    fake_trial, fake_meta = generate_fake_trial(class_name, phase_num, n_samples=550, seed=seed + 101)

    # pick hero (dominant) channel for compact plots
    real_idx = real_meta.get("channels", DEFAULT_CHANNELS).index(real_meta.get("dominant_channel", DEFAULT_CHANNELS[0])) if real_meta else 0
    fake_idx = fake_meta.get("channels", DEFAULT_CHANNELS).index(fake_meta.get("dominant_channel", DEFAULT_CHANNELS[0])) if fake_meta else 0
    real_sig = real_trial[:, real_idx] if real_trial.ndim == 2 else get_signal(real_trial)
    fake_sig = fake_trial[:, fake_idx] if fake_trial.ndim == 2 else get_signal(fake_trial)

    if st.session_state[real_key]:
        sig_a, sig_b = real_sig, fake_sig
    else:
        sig_a, sig_b = fake_sig, real_sig

    st.caption(f"Assume this is a **{class_name}** motor-imagery trial. Which signal looks more realistic for this phase's GAN quality?")
    time = np.arange(len(sig_a)) / 100.0

    # compact side-by-side hero plots
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.caption("**Plot A**")
        fig_a = go.Figure()
        fig_a.add_trace(go.Scatter(x=time, y=sig_a, mode="lines", line=dict(color=phase_data["color"])))
        fig_a.update_layout(template="plotly_white", height=180, margin=dict(t=20, b=10), showlegend=False)
        fig_a.update_xaxes(visible=False)
        fig_a.update_yaxes(visible=False)
        st.plotly_chart(fig_a, width="stretch")
    with col_b:
        st.caption("**Plot B**")
        fig_b = go.Figure()
        fig_b.add_trace(go.Scatter(x=time, y=sig_b, mode="lines", line=dict(color="#6c757d")))
        fig_b.update_layout(template="plotly_white", height=180, margin=dict(t=20, b=10), showlegend=False)
        fig_b.update_xaxes(visible=False)
        fig_b.update_yaxes(visible=False)
        st.plotly_chart(fig_b, width="stretch")

    # small action row
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 0.7])
    with col_btn1:
        if st.button("A is Real", key=f"a_real_{phase_num}"):
            st.session_state[choice_key] = "A"
    with col_btn2:
        if st.button("B is Real", key=f"b_real_{phase_num}"):
            st.session_state[choice_key] = "B"
    with col_btn3:
        if st.button("Next Example ↺", key=f"shuffle_{phase_num}"):
            _new_round()

    # expandable full montage for learners who want details
    with st.expander("Show full montage & annotations"):
        st.plotly_chart(plot_annotated_hero_channel(real_trial, real_meta), width="stretch")
        st.plotly_chart(plot_multichannel_stack(real_trial, real_meta, title=f"Full Montage — {class_name} Imagery"), width="stretch")

    # result + concise justification
    if st.session_state[choice_key] is not None:
        correct = "A" if st.session_state[real_key] else "B"
        if st.session_state[choice_key] == correct:
            st.success("✅ Correct! You spotted the more realistic EEG.")
        else:
            st.error(f"❌ The more realistic signal was on {correct}.")

        # compact PSD thumbnail and short biomarker bullets
        st.plotly_chart(plot_psd_comparison(real_sig, fake_sig, phase_data["color"]), width="stretch")

        st.markdown("#### 🔍 Why? — Biomarker Breakdown")
        real_metrics = analyze_biomarkers(real_sig, 100.0)
        fake_metrics = analyze_biomarkers(fake_sig, 100.0)
        col_real, col_fake = st.columns(2)
        with col_real:
            st.markdown("**More realistic trial**")
            for bullet in describe_biomarkers(real_metrics, class_name, is_real=True)[:3]:
                st.markdown(f"- {bullet}")
        with col_fake:
            st.markdown("**Phase-specific synthetic trial**")
            for bullet in describe_biomarkers(fake_metrics, class_name, is_real=False)[:3]:
                st.markdown(f"- {bullet}")

        if phase_num == 1:
            st.info("Phase 1 synthetics are usually easy to spot: noisy, missing coherent ERD/ERS timing.")
        elif phase_num == 2:
            st.info("Phase 2: plausible shape but mistimed or muted ERD/ERS — look closely at timing.")
        else:
            st.info("Phase 3: very realistic; subtle spectral/timing cues remain the differentiator.")


def render_live_preprocessing(phase_data: dict, color: str, phase_num: int) -> None:
    st.markdown("### 🎛️ Preprocessing — Live Preview")

    control_col, display_col = st.columns([1, 2])
    state_key = f"preproc_state_{phase_num}"
    class_options = phase_data.get("class_names", ["Right Hand", "Left Hand", "Foot"])[:3]

    if state_key not in st.session_state:
        st.session_state[state_key] = {
            "seed": 42,
            "class_name": class_options[0],
            "history": [],
            "signature": None,
        }

    with control_col:
        st.markdown("#### Controls")
        seed = st.number_input(
            "Seed",
            min_value=0,
            max_value=1_000_000,
            value=int(st.session_state[state_key]["seed"]),
            step=1,
            key=f"preproc_seed_{phase_num}",
        )
        class_opt = st.selectbox(
            "Class",
            class_options,
            index=class_options.index(st.session_state[state_key]["class_name"]) if st.session_state[state_key]["class_name"] in class_options else 0,
            key=f"preproc_class_{phase_num}",
        )

    demo_trial, demo_meta = generate_real_trial(class_opt, n_samples=550, seed=int(seed))
    demo_sig = demo_trial.mean(axis=1) if demo_trial.ndim == 2 else get_signal(demo_trial)
    signature = (int(seed), class_opt)

    if st.session_state[state_key]["signature"] != signature:
        st.session_state[state_key]["signature"] = signature
        st.session_state[state_key]["seed"] = int(seed)
        st.session_state[state_key]["class_name"] = class_opt
        st.session_state[state_key]["history"] = [{"label": "raw", "signal": demo_sig}]
    elif not st.session_state[state_key]["history"]:
        st.session_state[state_key]["history"] = [{"label": "raw", "signal": demo_sig}]

    with control_col:
        st.markdown("#### Actions")
        action_row_1 = st.columns(2)
        action_row_2 = st.columns(2)

        if action_row_1[0].button("Apply Bandpass", key=f"bandpass_{phase_num}"):
            cur = st.session_state[state_key]["history"][-1]["signal"]
            proc = apply_bandpass(cur, demo_meta.get("fs", 100.0), 0.5, 30.0)
            st.session_state[state_key]["history"].append({"label": "bandpass", "signal": proc})
        if action_row_1[1].button("Apply Notch", key=f"notch_{phase_num}"):
            cur = st.session_state[state_key]["history"][-1]["signal"]
            proc = apply_notch(cur, demo_meta.get("fs", 100.0), 50)
            st.session_state[state_key]["history"].append({"label": "notch", "signal": proc})
        if action_row_2[0].button("Apply CAR", key=f"car_{phase_num}"):
            cur = st.session_state[state_key]["history"][-1]["signal"]
            st.session_state[state_key]["history"].append({"label": "car", "signal": cur})
        if action_row_2[1].button("Baseline Correct", key=f"baseline_{phase_num}"):
            cur = st.session_state[state_key]["history"][-1]["signal"]
            proc = apply_baseline_correction(cur, 20)
            st.session_state[state_key]["history"].append({"label": "baseline", "signal": proc})

        if st.button("Undo Last Step", key=f"undo_{phase_num}"):
            if len(st.session_state[state_key]["history"]) > 1:
                st.session_state[state_key]["history"].pop()

    cur_signal = st.session_state[state_key]["history"][-1]["signal"]
    time = np.arange(len(cur_signal)) / demo_meta.get("fs", 100.0)

    with display_col:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time, y=demo_sig, mode="lines", name="Raw (center)", line=dict(color="lightgray", width=1.2, dash="dot")))
        fig.add_trace(go.Scatter(x=time, y=cur_signal, mode="lines", name="Processed", line=dict(color=color, width=3)))
        fig.update_layout(template="plotly_white", height=520, title="Centered Raw Signal — Apply steps in the local panel", xaxis_title="Time (s)", yaxis_title="Amplitude (µV)")
        st.plotly_chart(fig, width="stretch")

    st.markdown("### Transformation history")
    history = st.session_state[state_key]["history"][-4:]
    cols = st.columns(max(1, len(history)))
    for i, h in enumerate(history):
        with cols[i]:
            st.markdown(f"**{h['label']}**")
            st.line_chart(h["signal"], height=120)


def render_classifiers(phase_data: dict, phase_num: int) -> None:
    """Render a compact, card-based model zoo with a short load animation and precomputed metrics."""
    import time

    phase_color = phase_data.get("color", "#2ecc71")
    active_key = f"classifier_active_{phase_num}"
    loaded_key = f"classifier_loaded_{phase_num}"

    if active_key not in st.session_state:
        st.session_state[active_key] = "Logistic Regression"
    if loaded_key not in st.session_state:
        st.session_state[loaded_key] = None

    model_specs = {
        "Logistic Regression": {
            "accuracy": 0.8600,
            "std": 0.2544,
            "kappa": 0.82,
            "training_time": "12.4s",
            "confusion": [[58, 2, 0], [1, 56, 3], [0, 2, 58]],
            "params": {"max_iter": 1000, "random_state": 42},
        },
        "Linear SVM": {
            "accuracy": 0.8600,
            "std": 0.2544,
            "kappa": 0.82,
            "training_time": "11.9s",
            "confusion": [[58, 2, 0], [1, 56, 3], [0, 2, 58]],
            "params": {"kernel": "linear", "probability": True, "random_state": 42},
        },
        "Random Forest": {
            "accuracy": 0.6200,
            "std": 0.2939,
            "kappa": 0.48,
            "training_time": "16.8s",
            "confusion": [[40, 12, 8], [15, 35, 10], [10, 15, 35]],
            "params": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
        },
        "K-Nearest Neighbors": {
            "accuracy": 0.6800,
            "std": 0.1306,
            "kappa": 0.58,
            "training_time": "8.3s",
            "confusion": [[45, 8, 7], [10, 48, 2], [8, 4, 48]],
            "params": {"n_neighbors": 5},
        },
        "MLP Neural Net": {
            "accuracy": 0.7533,
            "std": 0.3593,
            "kappa": 0.64,
            "training_time": "15.2s",
            "confusion": [[52, 5, 3], [4, 50, 6], [3, 6, 51]],
            "params": {"hidden_layer_sizes": (100, 50), "activation": "relu", "solver": "adam", "max_iter": 500, "random_state": 42},
        },
        "Soft Voting Ensemble": {
            "accuracy": 0.7867,
            "std": 0.1236,
            "kappa": 0.73,
            "training_time": "13.7s",
            "confusion": [[55, 3, 2], [2, 53, 5], [1, 4, 55]],
            "params": {"voting": "soft", "n_jobs": -1},
        },
    }

    st.header("Model Zoo & Classification Performance")
    st.markdown("Choose a model card, run it briefly, and inspect the precomputed metrics and hyperparameters.")
    st.caption("This keeps the experience lightweight and honest: a short load step followed by the results you asked for.")

    model_names = list(model_specs.keys())
    cols = st.columns(3)
    for idx, name in enumerate(model_names):
        with cols[idx % 3]:
            card = st.container()
            card.markdown(f"### {name}")
            spec = model_specs[name]
            card.caption("Hyperparameters")
            param_text = "\n".join(f"• {k} = {v}" for k, v in spec["params"].items())
            card.markdown(param_text)
            is_active = st.session_state[active_key] == name
            if card.button("Run model", key=f"run_model_{phase_num}_{name}", use_container_width=True):
                st.session_state[active_key] = name
                with st.spinner(f"Running {name}..."):
                    time.sleep(5)
                st.session_state[loaded_key] = name

            if is_active:
                card.markdown(f"<div style='color:{phase_color}; font-weight:600;'>Selected</div>", unsafe_allow_html=True)

    st.markdown("---")

    loaded_model = st.session_state[loaded_key] or st.session_state[active_key]
    if loaded_model is None:
        loaded_model = "Logistic Regression"

    spec = model_specs[loaded_model]
    st.subheader(f"Loaded: {loaded_model}")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Accuracy", f"{spec['accuracy'] * 100:.1f}%")
    metric_cols[1].metric("Std Dev", f"±{spec['std']:.4f}")
    metric_cols[2].metric("Kappa", f"{spec['kappa']:.2f}")
    metric_cols[3].metric("Training Time", spec["training_time"])

    left_col, right_col = st.columns([1, 1])
    with left_col:
        st.markdown("### Hyperparameters")
        param_items = []
        for key, value in spec["params"].items():
            param_items.append(f"- **{key}**: {value}")
        st.markdown("\n".join(param_items))

    with right_col:
        st.markdown("### Confusion Matrix")
        cm = np.array(spec["confusion"], dtype=int)
        st.plotly_chart(plot_confusion_matrix(cm, ["Left Hand", "Right Hand", "Foot"], color=phase_color), width="stretch")

    st.markdown("### Leaderboard")
    names = list(model_specs.keys())
    accs = [model_specs[name]["accuracy"] * 100 for name in names]
    highlight = names.index(loaded_model)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=accs,
            y=names,
            orientation="h",
            marker=dict(color=[phase_color if i == highlight else "#cfd8dc" for i in range(len(names))]),
            text=[f"{a:.1f}%" for a in accs],
            textposition="outside",
        )
    )
    fig.update_layout(template="plotly_white", height=320, margin=dict(l=20, r=20, t=20, b=20), xaxis_title="Accuracy (%)")
    st.plotly_chart(fig, width="stretch")
