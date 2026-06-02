"""Data augmentation for Wi‑Fi CSI fall detection.

Implements the four augmentations from the paper Section III-B:

1. **Time Stretching** (III-B-a, Eq. 3) — stretch/compress each subcarrier
   independently with s ∈ [0.5, 2.5], then interpolate back to 625 steps.
2. **Gaussian Noise** (III-B-b) — zero‑mean additive noise with
   σ ∈ [0.1, 0.8].
3. **Temporal Shadowing** (III-B-c) — 3 segments, keep 150‑step window
   from each, concat 450 → interpolate back to 625. Two augmented samples
   per original.
4. **Asymmetric Signal Mixing** (III-B-d, Eq. 4) — same‑class pair,
   α ∈ [0.1, 0.3], (1-α)*x1 + α*x2. Fig. 4: ~72%/28% split.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════
# 1. Time Stretching  (paper Section III-B-a, Eq. 3)
# ═══════════════════════════════════════════════════════════════════════

def time_stretching(
    x: torch.Tensor,
    s_min: float = 0.7,
    s_max: float = 1.5,
    full_len: int = 625,
) -> torch.Tensor:
    """Stretch or compress each subcarrier independently along the time axis.

    Paper: ``s ∈ [0.5, 2.5]`` applied per‑subcarrier (Eq. 3).

    Args:
        x: ``[..., T, S]`` — T=625 time steps, S=90 subcarriers.
           The operation acts on dim=-2 (time).
        s_min, s_max: stretch factor range.
        full_len: target output length (always 625).

    Returns:
        Tensor with same shape as *x*.
    """
    original_shape = x.shape
    x_work = x.unsqueeze(0) if x.ndim == 2 else x  # ensure [C, T, S]
    C, T, S = x_work.shape

    out_parts = []
    for ch in range(S):
        s = random.uniform(s_min, s_max)
        new_len = max(1, int(T * s))
        # Interpolate channel data to new_len
        ch_data = x_work[:, :, ch].unsqueeze(1)  # [C, 1, T]
        stretched = F.interpolate(
            ch_data, size=new_len, mode="linear", align_corners=False
        )  # [C, 1, new_len]
        # Interpolate back to full_len
        restored = F.interpolate(
            stretched, size=full_len, mode="linear", align_corners=False
        )  # [C, 1, 625]
        out_parts.append(restored.squeeze(1))  # [C, 625]

    out = torch.stack(out_parts, dim=-1)  # [C, 625, S]

    if x.ndim == 2:
        out = out.squeeze(0)
    return out.to(device=x.device, dtype=x.dtype)


def apply_stretching_to_batch(
    batch_x: torch.Tensor,
    p_stretch: float = 0.5,
) -> torch.Tensor:
    out = batch_x.clone()
    for i in range(batch_x.shape[0]):
        if random.random() < p_stretch:
            out[i] = time_stretching(batch_x[i])
    return out


# ═══════════════════════════════════════════════════════════════════════
# 2. Gaussian Noise  (paper Section III-B-b)
# ═══════════════════════════════════════════════════════════════════════

def gaussian_noise(
    x: torch.Tensor,
    sigma_min: float = 0.05,
    sigma_max: float = 0.3,
) -> torch.Tensor:
    """Add zero‑mean Gaussian noise with σ ∈ [0.1, 0.8].

    Paper: additive white Gaussian noise drawn per sample (III-B-b).
    """
    sigma = random.uniform(sigma_min, sigma_max)
    noise = torch.randn_like(x) * sigma
    return x + noise


def apply_noise_to_batch(
    batch_x: torch.Tensor,
    p_noise: float = 0.5,
) -> torch.Tensor:
    out = batch_x.clone()
    for i in range(batch_x.shape[0]):
        if random.random() < p_noise:
            out[i] = gaussian_noise(batch_x[i])
    return out


# ═══════════════════════════════════════════════════════════════════════
# 3. Asymmetric Signal Mixing  (paper Section III-B-d, Eq. 4)
# ═══════════════════════════════════════════════════════════════════════

def asymmetric_signal_mixing(
    x1: torch.Tensor,
    x2: torch.Tensor,
    alpha_min: float = 0.1,
    alpha_max: float = 0.3,
) -> torch.Tensor:
    """Mix two CSI tensors asymmetrically — paper Eq. 4.

    α ∈ [0.1, 0.3]. The mixed sample stays closer to x1 because
    (1-α) ≥ 0.7 dominates.  Paper Fig. 4 shows ~72%/28% split.

    Args:
        x1, x2: CSI tensors of shape ``[625, 90]`` or ``[1, 625, 90]``.
        alpha_min, alpha_max: range for α (paper: 0.1–0.3).

    Returns:
        ``(1 - α) * x1 + α * x2``
    """
    alpha = random.uniform(alpha_min, alpha_max)
    return (1.0 - alpha) * x1 + alpha * x2


def apply_mixing_to_batch(
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    p_mix: float = 0.5,
    alpha_range: tuple[float, float] = (0.1, 0.3),
) -> torch.Tensor:
    """Apply asymmetric mixing **within** a batch.

    For each sample selected for mixing (probability *p_mix*), a partner
    with the **same label** is randomly chosen from the batch.  If no
    same‑class partner exists the sample is left unchanged.

    Args:
        batch_x: ``[N, 1, 625, 90]`` or ``[N, 625, 90]``.
        batch_y: ``[N]`` int labels (0 = non‑fall, 1 = fall).
        p_mix: probability of applying mixing per sample.
        alpha_range: (min, max) for α.

    Returns:
        Augmented batch tensor (same shape as *batch_x*).
    """
    N = batch_x.shape[0]
    out = batch_x.clone()
    y_np = batch_y.cpu().numpy() if isinstance(batch_y, torch.Tensor) else np.asarray(batch_y)

    # Index samples by class
    class_indices: dict[int, list[int]] = {}
    for i in range(N):
        label = int(y_np[i])
        class_indices.setdefault(label, []).append(i)

    alpha_min, alpha_max = alpha_range

    for i in range(N):
        if random.random() >= p_mix:
            continue
        label = int(y_np[i])
        partners = class_indices.get(label, [])
        if len(partners) < 2:
            continue
        # Pick a partner that is not self
        j = random.choice([p for p in partners if p != i])
        out[i] = asymmetric_signal_mixing(batch_x[i], batch_x[j], alpha_min, alpha_max)

    return out


# ═══════════════════════════════════════════════════════════════════════
# Temporal Shadowing
# ═══════════════════════════════════════════════════════════════════════

_SEGMENTS = 3
_WINDOW_LEN = 150
_TARGET_LEN = 450
_FULL_LEN = 625


def temporal_shadowing(
    x: torch.Tensor,
    window_len: int = _WINDOW_LEN,
    target_len: int = _TARGET_LEN,
    full_len: int = _FULL_LEN,
) -> torch.Tensor:
    """Apply temporal shadowing to a single CSI sample.

    Algorithm (from the paper):
      1. Divide the 625‑step time axis into 3 equal segments:
         [0-208], [209-416], [417-624].
      2. From each segment randomly extract a contiguous 150‑step window.
      3. Concatenate the 3 windows → 450‑step signal.
      4. Linearly interpolate back to 625 steps.

    Args:
        x: ``[625, 90]`` or ``[1, 625, 90]`` or ``[C, 625, 90]``.
           The augmentation operates on dim=-2 (time axis).

    Returns:
        Tensor with same shape as *x*, time axis restored to 625.
    """
    original_shape = x.shape
    was_batched = x.ndim == 3

    if not was_batched:
        x_work = x.unsqueeze(0)  # [1, 625, 90]
    else:
        x_work = x

    C, T, S = x_work.shape
    seg_len = T // _SEGMENTS  # 625 // 3 = 208

    windows: list[torch.Tensor] = []
    for seg_idx in range(_SEGMENTS):
        seg_start = seg_idx * seg_len
        seg_end = seg_start + seg_len if seg_idx < _SEGMENTS - 1 else T
        # Ensure there is room for the window
        max_start = max(seg_start, seg_end - window_len)
        if max_start <= seg_start:
            win_start = seg_start
        else:
            win_start = random.randint(seg_start, max_start)
        win = x_work[:, win_start : win_start + window_len, :]  # [C, 150, 90]
        windows.append(win)

    # Concatenate along time axis → [C, 450, 90]
    shadowed = torch.cat(windows, dim=1)

    # Interpolate back to 625
    # Permute to [C, 90, 450] for interpolate, then back
    shadowed_p = shadowed.permute(0, 2, 1)  # [C, 90, 450]
    restored_p = F.interpolate(
        shadowed_p,
        size=full_len,
        mode="linear",
        align_corners=False,
    )  # [C, 90, 625]
    restored = restored_p.permute(0, 2, 1)  # [C, 625, 90]

    if not was_batched:
        restored = restored.squeeze(0)

    return restored.to(device=x.device, dtype=x.dtype)


def apply_shadowing_to_batch(
    batch_x: torch.Tensor,
    p_shadow: float = 0.5,
) -> torch.Tensor:
    """Apply temporal shadowing to each sample in *batch_x* with prob *p_shadow*.

    Args:
        batch_x: ``[N, 1, 625, 90]`` or ``[N, 625, 90]``.
        p_shadow: per‑sample application probability.

    Returns:
        Augmented batch (same shape).
    """
    out = batch_x.clone()
    for i in range(batch_x.shape[0]):
        if random.random() < p_shadow:
            out[i] = temporal_shadowing(batch_x[i])
    return out


# ═══════════════════════════════════════════════════════════════════════
# Combined augmentation pipeline (for DataLoader integration)
# ═══════════════════════════════════════════════════════════════════════

class CSIAugmentation:
    """Apply all four paper augmentations with configurable probabilities.

    Paper Section III-B: time stretching, Gaussian noise, temporal
    shadowing, and asymmetric signal mixing.

    Usage inside a Dataset or collate function::

        aug = CSIAugmentation(p_mix=0.5, p_shadow=0.5, p_stretch=0.5, p_noise=0.5)
        batch_x, batch_y = next(iter(dataloader))
        batch_x = aug(batch_x, batch_y)
    """

    def __init__(
        self,
        p_mix: float = 0.5,
        p_shadow: float = 0.5,
        p_stretch: float = 0.5,
        p_noise: float = 0.5,
        alpha_range: tuple[float, float] = (0.1, 0.3),
        shadow_window_len: int = _WINDOW_LEN,
    ) -> None:
        self.p_mix = p_mix
        self.p_shadow = p_shadow
        self.p_stretch = p_stretch
        self.p_noise = p_noise
        self.alpha_range = alpha_range
        self.shadow_window_len = shadow_window_len

    def __call__(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
    ) -> torch.Tensor:
        """Apply augmentations in paper order and return augmented batch."""
        # 1. Time stretching (III-B-a)
        batch_x = apply_stretching_to_batch(batch_x, p_stretch=self.p_stretch)
        # 2. Gaussian noise (III-B-b)
        batch_x = apply_noise_to_batch(batch_x, p_noise=self.p_noise)
        # 3. Temporal shadowing (III-B-c)
        batch_x = apply_shadowing_to_batch(batch_x, p_shadow=self.p_shadow)
        # 4. Asymmetric signal mixing (III-B-d, Eq. 4)
        batch_x = apply_mixing_to_batch(
            batch_x, batch_y, p_mix=self.p_mix, alpha_range=self.alpha_range
        )
        return batch_x
