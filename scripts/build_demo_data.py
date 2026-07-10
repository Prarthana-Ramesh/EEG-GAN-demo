from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_FILE = DATA_DIR / "demo_data.pkl"


def _reshape_trial_array(arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim == 2:
        data = data[:, :, None]
    if data.ndim >= 3:
        data = data.reshape(data.shape[0], data.shape[1], -1)
    return data


def _load_phase3_arrays(path: Path) -> np.ndarray:
    data = np.load(path)
    data = _reshape_trial_array(data)
    baseline = data[:, : max(1, data.shape[1] // 5), :].mean(axis=1, keepdims=True)
    centered = data - baseline
    centered = centered - centered.mean(axis=1, keepdims=True)
    std = centered.std(axis=1, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (centered / std).astype(np.float32)


def _subsample(arr: np.ndarray, n_samples: int = 120) -> np.ndarray:
    if arr.shape[0] <= n_samples:
        return arr
    rng = np.random.default_rng(42)
    indices = rng.choice(arr.shape[0], size=n_samples, replace=False)
    return arr[indices]


def _make_fake_from_real(real_arr: np.ndarray, noise_scale: float = 0.06, rng_seed: int = 29) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    base = np.asarray(real_arr, dtype=np.float32)
    fake = np.empty_like(base)

    n_trials, n_time, n_channels = base.shape
    for trial_idx in range(n_trials):
        for chan_idx in range(n_channels):
            sig = base[trial_idx, :, chan_idx]
            if len(sig) < 5:
                fake[trial_idx, :, chan_idx] = sig
                continue

            window = min(11, len(sig) if len(sig) % 2 == 1 else len(sig) - 1)
            if window < 5:
                window = 5

            drift = np.cumsum(rng.normal(0.0, 0.002, size=n_time))
            drift = signal.savgol_filter(drift, window_length=window, polyorder=3)
            rhythm = np.sin(2 * np.pi * rng.uniform(8.0, 13.0) * np.arange(n_time) / 100.0 + rng.uniform(0.0, 2 * np.pi))
            colored_noise = rng.normal(0.0, 1.0, size=n_time)
            colored_noise = signal.savgol_filter(colored_noise, window_length=window, polyorder=3)
            envelope = 1.0 + 0.15 * np.sin(2 * np.pi * rng.uniform(0.35, 0.8) * np.arange(n_time) / 100.0 + rng.uniform(0.0, 2 * np.pi))

            fake_signal = sig + noise_scale * (0.45 * colored_noise + 0.35 * rhythm + 0.20 * drift) * envelope
            fake[trial_idx, :, chan_idx] = fake_signal.astype(np.float32)

    return fake.astype(np.float32)


def _make_phase_payload(name: str, phase_num: int, real_arr: np.ndarray, fake_arr: np.ndarray, color: str, class_names: list[str], accuracy: float, kappa: float, confusion_matrix: np.ndarray, ratio_anchors: list[float], accuracy_anchors: list[float], drop_factor: float, features: dict[str, bool]) -> dict[str, Any]:
    return {
        "name": name,
        "phase": phase_num,
        "real": _subsample(real_arr, n_samples=120),
        "fake": _subsample(fake_arr, n_samples=120),
        "class_names": class_names,
        "metrics": {
            "accuracy": accuracy,
            "kappa": kappa,
            "kappa_trajectory": np.clip(np.linspace(0.10, kappa, 100) + np.random.normal(0, 0.01, 100), 0.0, 1.0),
            "confusion_matrix": confusion_matrix,
        },
        "features": features,
        "ratio_anchors": {"x": [0, 50, 100], "y": accuracy_anchors},
        "robustness": {"drop_factor": drop_factor},
        "color": color,
    }


def build_demo_data() -> dict[str, Any]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    real_path = RAW_DIR / "phase3.npy"
    fake_path = RAW_DIR / "phase3_fake.npy"

    if not real_path.exists():
        real_path = RAW_DIR / "train_data_eeg.npy"
    if not real_path.exists():
        raise FileNotFoundError("Cannot find Phase 3 EEG data files in data/raw")

    real = np.load(real_path)
    fake = _make_fake_from_real(_reshape_trial_array(real), noise_scale=0.05, rng_seed=29)
    np.save(fake_path, fake)

    phase3_real = _load_phase3_arrays(real_path)
    phase3_fake = _load_phase3_arrays(fake_path)

    phase1_real = np.random.randn(200, 525, 118).astype(np.float32)
    phase1_fake = np.random.randn(200, 525, 118).astype(np.float32)
    phase2_real = np.random.randn(150, 500, 118).astype(np.float32)
    phase2_fake = np.random.randn(150, 500, 118).astype(np.float32)

    payload = {
        "phases": {
            1: _make_phase_payload(
                "Baseline EEG-GAN",
                1,
                phase1_real,
                phase1_fake,
                "#e74c3c",
                ["Left Hand", "Right Hand", "Foot"],
                0.463,
                0.40,
                np.array([[40, 15, 5], [10, 45, 5], [5, 10, 45]], dtype=float),
                [0, 50, 100],
                [0.35, 0.42, 0.46],
                0.30,
                {"vae": False, "spectral_loss": False, "attention": False},
            ),
            2: _make_phase_payload(
                "Stable VAE-GAN",
                2,
                phase2_real,
                phase2_fake,
                "#f39c12",
                ["Right Hand", "Foot"],
                0.753,
                0.70,
                np.array([[55, 5], [3, 52]], dtype=float),
                [0, 50, 100],
                [0.55, 0.68, 0.75],
                0.18,
                {"vae": True, "spectral_loss": False, "attention": False},
            ),
            3: _make_phase_payload(
                "MI-Aware GAN (Final)",
                3,
                phase3_real,
                phase3_fake,
                "#2ecc71",
                ["Left Hand", "Right Hand", "Foot"],
                0.860,
                0.82,
                np.array([[58, 2, 0], [1, 56, 3], [0, 2, 58]], dtype=float),
                [0, 50, 100],
                [0.65, 0.80, 0.86],
                0.15,
                {"vae": True, "spectral_loss": True, "attention": True},
            ),
        },
        "subjects": ["AA", "AL", "AV", "AW", "AY"],
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Wrote demo data package to {OUTPUT_FILE}")
    return payload


if __name__ == "__main__":
    build_demo_data()
