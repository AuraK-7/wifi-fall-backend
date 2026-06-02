"""Lightweight 2D-CNN fall detector — the paper's proposed architecture.

*Enhancing Deep Learning-Based Fall Detection Under Non-Dominant Paths Using Wi-Fi CSI*

Architecture (≈ 0.24 M parameters):
  Conv2D(1→32, 3×3) → BN → LeakyReLU → MaxPool(2,2)
  Conv2D(32→64, 3×3) → BN → LeakyReLU → MaxPool(2,2)
  Conv2D(64→128, 3×3) → BN → LeakyReLU → MaxPool(2,2)
  AdaptiveAvgPool2d(3,3) → Flatten
  FC(1152→128) + Dropout(0.5) + ReLU
  FC(128→1) → Sigmoid

Input:  [N, 1, 625, 90]  — CSI amplitude matrix (time × subcarriers)
Output: [N, 1]            — fall probability
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import numpy as np

from app.core.config import settings
from app.schemas.csi import CsiFrame, DetectionResult


class LightweightFallCNN(nn.Module):
    """Paper's lightweight 2D-CNN for Wi‑Fi CSI fall detection."""

    def __init__(self, dropout: float = 0.5) -> None:
        """Paper Section III-C: dropout 50% after dense layer (Fig. 5)."""
        super().__init__()

        self.features = nn.Sequential(
            # Block 1  — [1, 625, 90] → [32, 312, 45]
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2  — [32, 312, 45] → [64, 156, 22]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3  — [64, 156, 22] → [128, 78, 11]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.pool = nn.AdaptiveAvgPool2d((3, 3))  # → [128, 3, 3] = 1152

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        # Sigmoid is applied during inference via torch.sigmoid() —
        # training uses BCEWithLogitsLoss for numerical stability.

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                if m.out_features == 1:
                    # fan_in=128 → N(0, 0.125) with fan_in mode
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                else:
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (no sigmoid). Use with BCEWithLogitsLoss."""
        return self.classifier(self.pool(self.features(x)))

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return fall probability in [0, 1]."""
        return torch.sigmoid(self.forward(x))


# ---------------------------------------------------------------------------
# Inference wrapper — compatible with the existing detector interface
# ---------------------------------------------------------------------------

class CNN2DFallDetector:
    """Drop-in replacement for ``ENetFallDetector`` using the paper's 2D-CNN.

    Differences from B0 detector:
      - Input: [1, 625, 90] single‑channel (not [3, 625, 30] 3‑channel)
      - Model: 0.24 M params (vs 5.3 M)
      - Preprocessing: per-subcarrier Z‑score (via ``CsiZScoreNormalizer``)
    """

    model_name = "lightweight_2dcnn_enetfall"
    class_names = ["non_fall", "fall"]
    input_shape = [1, 625, 90]

    def __init__(
        self,
        model_path: str = "",
        normalizer_dir: str = "",
        device_str: str | None = None,
    ) -> None:
        self.model_path = model_path or settings.CNN2D_MODEL_PATH
        self.normalizer_dir = normalizer_dir or settings.CNN2D_NORMALIZER_DIR
        self.device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: LightweightFallCNN | None = None
        self.load_error: str | None = None

        # Lazy-import to avoid circular dependency at module level
        from app.services.csi_preprocessor import CsiZScoreNormalizer
        self._normalizer: CsiZScoreNormalizer | None = None
        try:
            self._normalizer = CsiZScoreNormalizer.load(self.normalizer_dir)
        except FileNotFoundError:
            pass  # will be reported in get_status()

        self._load_model()
        self._nlos_threshold: float = 0.85
        self._default_threshold: float = 0.70

    # ------------------------------------------------------------------
    @property
    def model_loaded(self) -> bool:
        return self.model is not None and self.load_error is None

    @property
    def normalizer_loaded(self) -> bool:
        return self._normalizer is not None and self._normalizer.is_fitted

    def reset(self) -> None:
        return None

    def get_status(self) -> dict[str, Any]:
        return {
            "detector_mode": "cnn2d",
            "model_loaded": self.model_loaded,
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": str(self.device),
            "num_classes": 2,
            "class_names": self.class_names,
            "input_shape": self.input_shape,
            "load_error": self.load_error,
            "normalizer_loaded": self.normalizer_loaded,
            "nlos_threshold": self._nlos_threshold,
            "default_threshold": self._default_threshold,
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(
        self,
        csi_2d: torch.Tensor,
        frame: CsiFrame,
    ) -> DetectionResult:
        """Run inference on a single [1, 625, 90] CSI window."""
        if self.model is None:
            raise RuntimeError(self.load_error or "2D-CNN model is not loaded")

        # Ensure 4D input: [1, 1, 625, 90]
        if csi_2d.dim() == 3:
            csi_2d = csi_2d.unsqueeze(0)  # [625, 90] → [1, 625, 90] → still 3D
        if csi_2d.dim() == 3:
            csi_2d = csi_2d.unsqueeze(0)  # [1, 625, 90] → [1, 1, 625, 90]

        # Apply Z‑score normalization
        if self._normalizer is not None:
            csi_2d = self._normalizer.normalize(csi_2d)

        tensor = csi_2d.to(self.device)
        self.model.eval()
        with torch.no_grad():
            prob_fall = float(self.model.predict_proba(tensor).item())

        prob_non_fall = 1.0 - prob_fall
        predicted_label = "fall" if prob_fall >= 0.5 else "non_fall"

        # Dynamic threshold: raise bar for NLoS rooms
        conf_threshold = self._get_confidence_threshold(frame.room, tensor)

        if predicted_label == "fall" and prob_fall >= conf_threshold:
            risk_level = "high"
            alert = True
        elif predicted_label == "fall":
            risk_level = "medium"
            alert = False
        else:
            risk_level = "low"
            alert = False

        return DetectionResult(
            timestamp=frame.timestamp,
            room=frame.room,
            predicted_label=predicted_label,
            confidence=round(prob_fall, 4),
            risk_level=risk_level,
            alert=alert,
            reason=(
                f"2D-CNN predicted {predicted_label} "
                f"(NLoS threshold={conf_threshold:.2f})"
            ),
            activity_score=round(prob_fall, 4),
            features={
                "model": self.model_name,
                "input_shape": self.input_shape,
                "prob_non_fall": round(prob_non_fall, 6),
                "prob_fall": round(prob_fall, 6),
                "threshold_used": conf_threshold,
                "true_label": frame.label or frame.simulated_label,
                "source": frame.source,
                "room": frame.room,
            },
        )

    def predict_from_numpy(
        self,
        data_2d: np.ndarray,
        frame: CsiFrame,
    ) -> DetectionResult:
        """Convenience wrapper: numpy [625, 90] → DetectionResult."""
        t = torch.from_numpy(data_2d.astype(np.float32)).unsqueeze(0)  # [1, 625, 90]
        return self.predict(t, frame)

    # ------------------------------------------------------------------
    # NLoS detection & dynamic threshold (P2)
    # ------------------------------------------------------------------
    _NLOS_ROOMS = {"home_lab_right", "home_lab(R)", "right_home_lab"}
    _NLOS_SIGNAL_VARIANCE_THRESHOLD = 0.005  # empirical, Z‑score normalised

    def _get_confidence_threshold(
        self,
        room: str,
        tensor: torch.Tensor | None = None,
    ) -> float:
        """Return a higher threshold for NLoS / through‑wall scenarios.

        Detection strategy:
          1. Room‑name based: if the room name contains known NLoS labels
          2. Signal‑based: if the subcarrier variance is unusually low
             (indicating attenuated through‑wall signal)
        """
        # Rule 1 — room name
        room_lower = room.lower().replace(" ", "_")
        if any(nlos in room_lower for nlos in self._NLOS_ROOMS):
            return self._nlos_threshold

        # Rule 2 — signal quality heuristic
        if tensor is not None:
            var = float(tensor.var().item())
            if var < self._NLOS_SIGNAL_VARIANCE_THRESHOLD:
                return self._nlos_threshold

        return self._default_threshold

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        model = LightweightFallCNN()
        path = Path(self.model_path)

        if not path.exists():
            self.model = model.to(self.device)
            self.model.eval()
            self.load_error = (
                f"2D‑CNN weights not found at {self.model_path}. "
                f"Using randomly-initialised model — run train.py first."
            )
            return

        try:
            state_dict = torch.load(path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.model = model
            self.load_error = None
        except Exception as exc:
            self.model = None
            self.load_error = str(exc)


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Quick sanity check
if __name__ == "__main__":
    m = LightweightFallCNN()
    n = count_parameters(m)
    print(f"LightweightFallCNN parameters: {n:,}  ({n / 1e6:.2f} M)")
    # Test forward pass
    x = torch.randn(4, 1, 625, 90)
    y = m(x)
    print(f"Input: {x.shape} → Output: {y.shape}  (values in [{y.min():.4f}, {y.max():.4f}])")
