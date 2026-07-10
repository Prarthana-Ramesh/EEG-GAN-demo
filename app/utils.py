from __future__ import annotations

import numpy as np
from scipy import signal


def get_signal(trial: np.ndarray) -> np.ndarray:
    """Safely extract a 1D time-series from any trial shape."""
    arr = np.asarray(trial, dtype=float)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        if arr.shape[1] == 1:
            return arr[:, 0]
        return arr.mean(axis=1)
    return np.mean(arr.reshape(arr.shape[0], -1), axis=1)


def ensure_2d(trial: np.ndarray) -> np.ndarray:
    """Ensure a trial is always shaped as (Time, Channels)."""
    arr = np.asarray(trial, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    return arr.reshape(arr.shape[0], -1)


def apply_bandpass(sig: np.ndarray, fs: float, lowcut: float, highcut: float, order: int = 4) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 1e-4) if lowcut > 0 else 1e-4
    high = min(highcut / nyq, 0.999) if highcut < fs / 2 else 0.999
    if low >= high or high <= 0 or low >= 1:
        return sig
    b, a = signal.butter(order, [low, high], btype="band")
    return signal.filtfilt(b, a, sig)


def apply_notch(sig: np.ndarray, fs: float, freq: float = 50.0, quality: float = 30.0) -> np.ndarray:
    """Apply a notch filter to remove line-noise components."""
    b, a = signal.iirnotch(freq, quality, fs)
    return signal.filtfilt(b, a, sig)


def apply_car(sig_2d: np.ndarray) -> np.ndarray:
    """Apply common-average re-referencing across channels."""
    arr = np.asarray(sig_2d, dtype=float)
    if arr.ndim == 1:
        return arr
    return arr - np.mean(arr, axis=1, keepdims=True)


def apply_baseline_correction(sig: np.ndarray, pre_cue_samples: int = 20) -> np.ndarray:
    """Subtract the pre-cue mean to zero the baseline."""
    baseline_mean = np.mean(sig[:pre_cue_samples]) if len(sig) >= pre_cue_samples else np.mean(sig)
    return sig - baseline_mean
