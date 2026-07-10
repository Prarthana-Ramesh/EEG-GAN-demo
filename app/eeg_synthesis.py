"""
eeg_synthesis.py
-----------------
Realistic synthetic EEG generation for the BCI Augmentation Pipeline demo.

This module builds physiologically-plausible motor-imagery (MI) EEG trials
from first principles (1/f background + band-limited mu/beta rhythms whose
amplitude is modulated by an ERD/ERS envelope + optional EOG/EMG artifacts),
instead of relying only on pre-computed pickle data.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal

# ----------------------------------------------------------------------
# Montage & class topography
# ----------------------------------------------------------------------

DEFAULT_CHANNELS = ["Fp1", "Fp2", "F3", "F4", "C3", "Cz", "C4", "Pz"]
FRONTAL_CHANNELS = {"Fp1", "Fp2", "F3", "F4"}
MOTOR_CHANNELS = {"C3", "Cz", "C4"}

CLASS_TOPOGRAPHY = {
    "Right Hand": {"C3": 1.00, "Cz": 0.30, "C4": 0.12, "F3": 0.10},
    "Left Hand": {"C4": 1.00, "Cz": 0.30, "C3": 0.12, "F4": 0.10},
    "Foot": {"Cz": 1.00, "C3": 0.35, "C4": 0.35},
    "Tongue": {"Cz": 0.55, "C3": 0.50, "C4": 0.50, "F3": 0.20, "F4": 0.20},
}

CLASS_DOMINANT_CHANNEL = {
    "Right Hand": "C3",
    "Left Hand": "C4",
    "Foot": "Cz",
    "Tongue": "Cz",
}

CLASS_COLORS = {
    "Right Hand": "#2ecc71",
    "Left Hand": "#3498db",
    "Foot": "#e67e22",
    "Tongue": "#9b59b6",
}

BASELINE_FRAC, MI_FRAC, REBOUND_FRAC = 0.18, 0.64, 0.18


# ----------------------------------------------------------------------
# Low-level signal building blocks
# ----------------------------------------------------------------------

def _pink_noise(n_samples: int, rng: np.random.Generator, exponent: float = 1.0) -> np.ndarray:
    """1/f^exponent background noise -- the dominant shape of real scalp EEG."""
    n_samples = max(int(n_samples), 8)
    white = rng.standard_normal(n_samples)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
    scale = 1.0 / np.power(freqs, exponent / 2.0)
    scale[0] = scale[1]
    pink = np.fft.irfft(spectrum * scale, n=n_samples)
    pink = pink / (np.std(pink) + 1e-9)
    return pink


def _band_oscillation(n_samples: int, fs: float, f_lo: float, f_hi: float,
                       rng: np.random.Generator) -> np.ndarray:
    """Band-limited quasi-rhythmic activity (mu, beta, ...), via bandpassed noise."""
    n_samples = max(int(n_samples), 8)
    white = rng.standard_normal(n_samples)
    nyq = fs / 2.0
    lo, hi = max(f_lo / nyq, 1e-4), min(f_hi / nyq, 0.99)
    try:
        sos = sp_signal.butter(4, [lo, hi], btype="bandpass", output="sos")
        padlen = min(3 * max(len(sos), 1) * 2, n_samples - 1) if n_samples > 8 else None
        band = sp_signal.sosfiltfilt(sos, white, padlen=padlen)
    except ValueError:
        t = np.arange(n_samples) / fs
        band = np.zeros(n_samples)
        for f in np.linspace(f_lo, f_hi, 4):
            band += np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    band = band / (np.std(band) + 1e-9)
    return band


def _erd_ers_envelope(n_samples: int, baseline_end: int, mi_end: int,
                       erd_depth: float, ers_gain: float,
                       smooth_samples: int = 5) -> np.ndarray:
    """Amplitude envelope: 1.0 at baseline, dips to (1-erd_depth) during MI,
    rises to (1+ers_gain) during the rebound window, smoothed at the edges."""
    env = np.ones(n_samples)
    env[baseline_end:mi_end] = 1.0 - erd_depth
    env[mi_end:] = 1.0 + ers_gain
    if smooth_samples > 1 and n_samples > smooth_samples * 2:
        kernel = np.ones(smooth_samples) / smooth_samples
        env = np.convolve(env, kernel, mode="same")
    return env


def _blink_pulse(n_samples: int, center: int, fs: float, amplitude: float) -> np.ndarray:
    """A single eye-blink transient: fast rise, slower exponential decay."""
    t = (np.arange(n_samples) - center) / fs
    pulse = np.where(t >= 0, np.exp(-t / 0.12), np.exp(t / 0.05))
    pulse = np.clip(pulse, 0, None)
    return amplitude * pulse


# ----------------------------------------------------------------------
# Trial generation
# ----------------------------------------------------------------------

def generate_mi_trial(
    class_name: str,
    n_samples: int = 550,
    fs: float = 100.0,
    channels: list[str] | None = None,
    quality: str = "real",
    seed: int | None = None,
    add_eog: bool = True,
    add_emg: bool | None = None,
) -> tuple[np.ndarray, dict]:
    """Generate one (n_samples, n_channels) synthetic MI trial."""
    channels = channels or DEFAULT_CHANNELS
    rng = np.random.default_rng(seed)
    n_ch = len(channels)
    topo = CLASS_TOPOGRAPHY.get(class_name, CLASS_TOPOGRAPHY["Right Hand"])
    if add_emg is None:
        add_emg = class_name in ("Foot", "Tongue")

    baseline_end = max(int(n_samples * BASELINE_FRAC), 3)
    mi_end = max(int(n_samples * (BASELINE_FRAC + MI_FRAC)), baseline_end + 3)
    mi_end = min(mi_end, n_samples - 2)

    if quality == "real" or quality == "phase3":
        erd_depth, ers_gain = 0.55, 0.45
        coherence = 1.0
        extra_noise = 0.12 if quality == "phase3" else 0.05
        smooth_extra = 0 if quality == "real" else 1
        bg_exponent = 1.0
    elif quality == "phase2":
        erd_depth, ers_gain = 0.25, 0.10
        coherence = 0.6
        extra_noise = 0.20
        smooth_extra = 2
        bg_exponent = 0.8
    else:
        erd_depth, ers_gain = 0.05, 0.02
        coherence = 0.15
        extra_noise = 0.9
        smooth_extra = 0
        bg_exponent = 0.3

    trial = np.zeros((n_samples, n_ch))
    eog_events: list[int] = []

    shared_mu = _band_oscillation(n_samples, fs, 8, 12, rng)
    shared_beta = _band_oscillation(n_samples, fs, 13, 30, rng)

    for ci, ch in enumerate(channels):
        bg = _pink_noise(n_samples, rng, exponent=bg_exponent) * 8.0

        own_mu = _band_oscillation(n_samples, fs, 8, 12, rng)
        own_beta = _band_oscillation(n_samples, fs, 13, 30, rng)
        mu = coherence * shared_mu + (1 - coherence) * own_mu
        beta = coherence * shared_beta + (1 - coherence) * own_beta

        weight = topo.get(ch, 0.05)
        env = _erd_ers_envelope(
            n_samples, baseline_end, mi_end,
            erd_depth=erd_depth * weight, ers_gain=ers_gain * weight,
            smooth_samples=5 + smooth_extra,
        )

        rhythm = env * (5.5 * mu + 3.0 * beta)
        extra = _pink_noise(n_samples, rng, exponent=0.2) * 6.0 * extra_noise

        ch_signal = bg + rhythm + extra

        if add_emg and ch in ("Cz", "F3", "F4") and quality in ("real", "phase3"):
            burst = _band_oscillation(n_samples, fs, 35, 90, rng)
            burst_env = np.zeros(n_samples)
            burst_env[mi_end:] = np.linspace(0, 1, n_samples - mi_end) ** 2
            ch_signal += burst * burst_env * 3.0

        if add_eog and ch in FRONTAL_CHANNELS:
            n_blinks = rng.integers(0, 2)
            frontal_gain = 1.0 if ch in ("Fp1", "Fp2") else 0.4
            for _ in range(n_blinks):
                center = int(rng.uniform(0, baseline_end * 0.9))
                amp = rng.uniform(25, 60) * frontal_gain
                ch_signal += _blink_pulse(n_samples, center, fs, amp)
                eog_events.append(center)
            drift = np.cumsum(rng.standard_normal(n_samples)) * 0.03 * frontal_gain
            ch_signal += drift - drift.mean()

        trial[:, ci] = ch_signal

    meta = {
        "fs": fs,
        "channels": channels,
        "class_name": class_name,
        "quality": quality,
        "n_samples": n_samples,
        "baseline_window": (0, baseline_end),
        "mi_window": (baseline_end, mi_end),
        "rebound_window": (mi_end, n_samples),
        "dominant_channel": CLASS_DOMINANT_CHANNEL.get(class_name, "C3"),
        "eog_events": sorted(set(eog_events)),
        "erd_depth": erd_depth,
        "ers_gain": ers_gain,
    }
    return trial, meta


def generate_real_trial(class_name: str, **kwargs) -> tuple[np.ndarray, dict]:
    return generate_mi_trial(class_name, quality="real", **kwargs)


def generate_fake_trial(class_name: str, phase_num: int, **kwargs) -> tuple[np.ndarray, dict]:
    quality = {1: "phase1", 2: "phase2", 3: "phase3"}.get(phase_num, "phase3")
    return generate_mi_trial(class_name, quality=quality, **kwargs)


# ----------------------------------------------------------------------
# Biomarker analysis (for the "why is this real/fake" explanations)
# ----------------------------------------------------------------------

def _bandpower(sig: np.ndarray, fs: float, f_lo: float, f_hi: float) -> float:
    sig = np.asarray(sig, dtype=float)
    if len(sig) < 8:
        return float(np.var(sig))
    freqs, psd = sp_signal.welch(sig, fs=fs, nperseg=min(len(sig), 64))
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(mask):
        return 0.0
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz_fn(psd[mask], freqs[mask]))


def _spectral_slope(sig: np.ndarray, fs: float) -> float:
    """Rough 1/f slope of the log-log PSD between 2-40 Hz."""
    sig = np.asarray(sig, dtype=float)
    if len(sig) < 16:
        return float("nan")
    freqs, psd = sp_signal.welch(sig, fs=fs, nperseg=min(len(sig), 128))
    mask = (freqs >= 2) & (freqs <= 40) & (psd > 0)
    if mask.sum() < 4:
        return float("nan")
    slope, _ = np.polyfit(np.log10(freqs[mask]), np.log10(psd[mask]), 1)
    return float(slope)


def analyze_biomarkers(sig_1d: np.ndarray, fs: float,
                        baseline_frac: float = BASELINE_FRAC,
                        mi_frac: float = MI_FRAC) -> dict:
    """Compute ERD%, ERS%, and spectral-slope diagnostics from a 1D signal."""
    sig_1d = np.asarray(sig_1d, dtype=float)
    n = len(sig_1d)
    b_end = max(int(n * baseline_frac), 3)
    m_end = min(max(int(n * (baseline_frac + mi_frac)), b_end + 3), n - 2)

    baseline = sig_1d[:b_end]
    mi = sig_1d[b_end:m_end]
    rebound = sig_1d[m_end:]

    base_power = _bandpower(baseline, fs, 8, 30) + 1e-9
    mi_power = _bandpower(mi, fs, 8, 30)
    rebound_power = _bandpower(rebound, fs, 8, 30) if len(rebound) > 8 else base_power

    erd_pct = (base_power - mi_power) / base_power * 100.0
    ers_pct = (rebound_power - base_power) / base_power * 100.0
    slope = _spectral_slope(sig_1d, fs)

    return {
        "erd_pct": erd_pct,
        "ers_pct": ers_pct,
        "spectral_slope": slope,
        "baseline_window": (0, b_end),
        "mi_window": (b_end, m_end),
        "rebound_window": (m_end, n),
    }


def describe_biomarkers(metrics: dict, class_name: str, is_real: bool) -> list[str]:
    """Plain-language bullet points explaining what the numbers mean."""
    erd, ers, slope = metrics["erd_pct"], metrics["ers_pct"], metrics["spectral_slope"]
    bullets = []

    if erd > 15:
        bullets.append(
            f"**ERD present:** mu/beta power drops about **{erd:.0f}%** during the "
            f"imagined-movement window — the expected desynchronization signature for {class_name.lower()} imagery."
        )
    elif erd > 0:
        bullets.append(
            f"**ERD weak:** mu/beta power only dips **{erd:.0f}%** during the movement window — "
            "real motor cortex usually desynchronizes more strongly than this."
        )
    else:
        bullets.append(
            "**No ERD:** mu/beta power doesn't drop during the movement window at all — "
            "real sensorimotor cortex almost always desynchronizes during motor imagery, so this is a red flag."
        )

    if ers > 15:
        bullets.append(
            f"**ERS / beta rebound present:** power rebounds **{ers:.0f}%** above baseline right after the "
            "movement window ends — the classic post-movement beta rebound."
        )
    elif ers > 0:
        bullets.append(f"**ERS weak:** only a **{ers:.0f}%** rebound after the movement window — muted post-movement recovery.")
    else:
        bullets.append("**No ERS:** there's no post-movement beta rebound — a common tell of lower-fidelity synthetic EEG.")

    if not np.isnan(slope):
        if -2.2 <= slope <= -0.7:
            bullets.append(f"**Spectrum shape (1/f slope ≈ {slope:.2f}):** matches the smooth 1/f falloff of real scalp EEG.")
        else:
            bullets.append(f"**Spectrum shape (1/f slope ≈ {slope:.2f}):** unusually flat/steep for real EEG — often means the noise floor doesn't match a real amplifier + brain.")

    verdict = "consistent with a **real** recording" if is_real else "the kind of pattern that gives away a **synthetic** trial"
    bullets.append(f"Taken together, this profile is {verdict}.")
    return bullets


__all__ = [
    "DEFAULT_CHANNELS",
    "CLASS_COLORS",
    "CLASS_DOMINANT_CHANNEL",
    "CLASS_TOPOGRAPHY",
    "generate_mi_trial",
    "generate_real_trial",
    "generate_fake_trial",
    "analyze_biomarkers",
    "describe_biomarkers",
]
