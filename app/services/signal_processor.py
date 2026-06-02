"""CSI signal processing — STFT, antenna correlation, spectral moments.

All functions accept a pre-processed torch tensor of shape [3, 625, 30]
(antennas, time-packets, subcarriers) and return a flat analytics dict
suitable for direct JSON serialisation into the WebSocket payload.

Perf target: < 5 ms per 100-ms tick on a modern CPU.
"""

from __future__ import annotations

import numpy as np
import scipy.signal
import torch

# STFT parameters 
STFT_FS = 100            # Hz  — sampling rate
STFT_NPERSEG = 64        # samples per segment
STFT_NOVERLAP = 48       # overlap
STFT_NFFT = 256          # FFT points


def compute_analytics(window: torch.Tensor) -> dict:
    """Compute analytics for one CSI window.

    Handles both input shapes:
      - [3, 625, 30]  — 3‑channel B0 format (antennas × time × subcarriers)
      - [1, 625, 90]  — single‑channel 2D‑CNN format
    """
    arr = window.detach().cpu().numpy().astype(np.float32)

    # Collapse all leading dims (antenna / channel) → [T, S] 2‑D matrix
    if arr.ndim == 3:
        # Average across the first (antenna/channel) dimension
        arr_2d = arr.mean(axis=0)  # [625, S]
    elif arr.ndim == 2:
        arr_2d = arr                # already [625, S]
    else:
        # Fallback: flatten all but last dim
        arr_2d = arr.reshape(-1, arr.shape[-1]).mean(axis=0, keepdims=False)
        if arr_2d.ndim == 1:
            arr_2d = arr_2d.reshape(1, -1)

    # 1‑D time series: average across subcarriers
    ts_1d = arr_2d.mean(axis=1)  # [625]

    f, _, Zxx = scipy.signal.stft(
        ts_1d,
        fs=STFT_FS,
        nperseg=STFT_NPERSEG,
        noverlap=STFT_NOVERLAP,
        nfft=STFT_NFFT,
    )
    spectrum_db = _to_db(np.abs(Zxx[1:, :]))
    centre_idx = spectrum_db.shape[1] // 2
    micro_doppler_spectrum = spectrum_db[:, centre_idx].tolist()

    # Antenna / subcarrier-band correlation
    if arr.ndim == 3 and arr.shape[0] == 3:
        # B0 format: true 3‑antenna spatial correlation
        antenna_correlation = _antenna_correlation(arr)
    else:
        # 2D‑CNN format: subcarrier-band coherence
        # Split subcarriers into two halves, correlate them across time
        S = arr_2d.shape[1]
        half = S // 2
        if half >= 4:
            a = arr_2d[:, :half].mean(axis=1)     # mean of first half  per time step
            b = arr_2d[:, half:].mean(axis=1)     # mean of second half per time step
            antenna_correlation = float(np.corrcoef(a, b)[0, 1])
        else:
            antenna_correlation = 0.0

    # Subcarrier amplitudes: last time slice of arr_2d
    subcarrier_amplitudes = arr_2d[-1, :].tolist()

    # Per-element mean energy — independent of window size, ~1.0 for Z-score data
    energy = float(np.mean(arr ** 2))
    signal_variance = float(np.var(subcarrier_amplitudes))

    spec = np.abs(Zxx[1:, centre_idx]).astype(np.float64)
    dominant_freq, frequency_spread = _spectral_moments(f[1:], spec)

    return {
        "micro_doppler_spectrum": [round(v, 4) for v in micro_doppler_spectrum],
        "subcarrier_amplitudes": [round(v, 6) for v in subcarrier_amplitudes],
        "antenna_correlation": round(antenna_correlation, 4),
        "energy": round(energy, 4),
        "dominant_freq": round(dominant_freq, 2),
        "frequency_spread": round(frequency_spread, 2),
        "signal_variance": round(signal_variance, 6),
    }



def _to_db(linear: np.ndarray, floor: float = -80.0) -> np.ndarray:
    """Convert linear magnitude to dB, clamped below at *floor*."""
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(linear + 1e-12)
    return np.clip(db, floor, 0.0)


def _antenna_correlation(arr: np.ndarray) -> float:
    """Mean pairwise Pearson correlation across the 3 antenna [30]-vectors.

    For each antenna we take the *last* time-slice (most recent packet)
    which gives a [30] vector per antenna, then compute the 3×3
    correlation matrix and return the mean of its upper triangle.
    """
    slices = arr[:, -1, :]  # [3, 30]
    corr = np.corrcoef(slices)  # [3, 3]
    iu = np.triu_indices(3, k=1)
    return float(corr[iu].mean())


def _spectral_moments(freqs: np.ndarray, spectrum: np.ndarray) -> tuple[float, float]:
    """Return (dominant_freq_Hz, frequency_spread_Hz) from a 1-D spectrum."""
    total = spectrum.sum()
    if total == 0:
        return 0.0, 0.0
    centroid = float(np.sum(freqs * spectrum) / total)
    spread = float(
        np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / total)
    )
    return centroid, spread
