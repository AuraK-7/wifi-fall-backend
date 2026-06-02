"""CSI data preprocessor — per-subcarrier Z-score normalization.

Replaces the ImageNet-based Normalize(mean=[0.485,0.456,0.406], ...) with
per-subcarrier Z‑score normalisation as specified in the paper:
  *Enhancing Deep Learning-Based Fall Detection Under Non-Dominant Paths
   Using Wi-Fi CSI*

The normaliser is fit on the training split only; statistics are cached to
disk so the inference pipeline can reload them without touching training data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch


class CsiZScoreNormalizer:
    """Per-subcarrier Z‑score normalisation for CSI amplitude tensors.

    Supports two input layouts:
      - ``[N, 625, 90]``  — raw 2‑D CSI (time × subcarriers)
      - ``[N, 3, 625, 30]`` — 3‑channel view (antenna-grouped)

    Statistics are computed along the **last dimension** (subcarrier axis),
    matching the paperʼs description of per-subcarrier normalisation.
    """

    STATS_FILENAME = "csi_zscore_stats.json"

    def __init__(
        self,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ) -> None:
        self.mean: np.ndarray | None = mean
        self.std: np.ndarray | None = std

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    @classmethod
    def fit_on_tensor(
        cls,
        data: torch.Tensor,
    ) -> "CsiZScoreNormalizer":
        """Compute per-subcarrier mean & std from a training tensor.

        ``data`` shape: ``[N, …, S]`` where *S* is the subcarrier dimension
        (90 for 2‑D layout, 30 for 3‑channel layout).
        """
        arr = data.numpy().astype(np.float64)
        # Collapse all dims *except* the last (subcarrier) one
        axes = tuple(range(arr.ndim - 1))
        mean = np.mean(arr, axis=axes).astype(np.float32)
        std = np.std(arr, axis=axes).astype(np.float32)
        std = np.where(std < 1e-8, 1.0, std)  # avoid div‑by‑zero
        return cls(mean=mean, std=std)

    @classmethod
    def fit_on_numpy(
        cls,
        data: np.ndarray,
    ) -> "CsiZScoreNormalizer":
        """Compute per-subcarrier mean & std from a numpy array.

        ``data`` shape: ``[N, …, S]``.
        """
        axes = tuple(range(data.ndim - 1))
        mean = np.mean(data, axis=axes).astype(np.float32)
        std = np.std(data, axis=axes).astype(np.float32)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean=mean, std=std)

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------
    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        """Apply Z‑score: ``(x - mean) / std`` broadcast along the last dim.

        Returns a **new** tensor; does not modify the input in-place.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("CsiZScoreNormalizer has not been fitted")

        device = data.device
        mean_t = torch.from_numpy(self.mean).to(device)
        std_t = torch.from_numpy(self.std).to(device)
        return (data - mean_t) / std_t

    def normalize_numpy(self, data: np.ndarray) -> np.ndarray:
        """NumPy version for pre-tensor preprocessing."""
        if self.mean is None or self.std is None:
            raise RuntimeError("CsiZScoreNormalizer has not been fitted")
        return (data - self.mean) / self.std

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, directory: str | Path) -> Path:
        """Persist mean & std as JSON in *directory*."""
        if self.mean is None or self.std is None:
            raise RuntimeError("Cannot save unfitted normalizer")
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        payload = {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }
        stats_path = dir_path / self.STATS_FILENAME
        stats_path.write_text(json.dumps(payload))
        return stats_path

    @classmethod
    def load(cls, directory: str | Path) -> "CsiZScoreNormalizer":
        """Load a previously saved normalizer from *directory*."""
        stats_path = Path(directory) / cls.STATS_FILENAME
        if not stats_path.exists():
            raise FileNotFoundError(
                f"Z‑score stats not found at {stats_path}. "
                f"Run the training script first."
            )
        payload = json.loads(stats_path.read_text())
        return cls(
            mean=np.array(payload["mean"], dtype=np.float32),
            std=np.array(payload["std"], dtype=np.float32),
        )

    @property
    def is_fitted(self) -> bool:
        return self.mean is not None and self.std is not None


def build_3channel_tensor(data_2d: np.ndarray) -> torch.Tensor:
    """Convert raw [N, 625, 90] CSI → [N, 3, 625, 30] 3‑channel tensor.

    Channels are formed by strided sub-sampling (step=3) as in the
    existing ``EnetFallMatDataSource._preprocess``, but **without**
    ImageNet normalisation — that step is handled separately by the
    ``CsiZScoreNormalizer``.
    """
    N = data_2d.shape[0]
    data_3ch = np.ndarray(shape=(N, 3, 625, 30), dtype=np.float32)
    data_3ch[:, 0, :, :] = data_2d[:, :, 0:90:3]
    data_3ch[:, 1, :, :] = data_2d[:, :, 1:90:3]
    data_3ch[:, 2, :, :] = data_2d[:, :, 2:90:3]
    return torch.from_numpy(data_3ch)


def build_2d_tensor(data_2d: np.ndarray) -> torch.Tensor:
    """Convert raw [N, 625, 90] CSI → [N, 1, 625, 90] single‑channel tensor.

    Used by the lightweight 2D‑CNN which expects the full CSI amplitude
    matrix as a 2‑D image.
    """
    return torch.from_numpy(data_2d.astype(np.float32)).unsqueeze(1)
