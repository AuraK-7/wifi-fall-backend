from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from app.core.config import settings
from app.schemas.csi import CsiFrame, DetectionResult


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TemporalConvTransformer(nn.Module):
    def __init__(
        self,
        input_features: int = 90,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
        max_len: int = 256,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_features, d_model, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.stem(x)
        seq = features.transpose(1, 2)
        if seq.shape[1] > self.pos_embed.shape[1]:
            seq = seq[:, : self.pos_embed.shape[1], :]
        seq = seq + self.pos_embed[:, : seq.shape[1], :]
        seq = self.encoder(seq)
        pooled = seq.mean(dim=1)
        return self.head(pooled)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


class TCNTransformerFallDetector:
    model_name = "tcn_transformer_enetfall"
    class_names = ["non_fall", "fall"]
    input_shape = [1, 625, 90]

    _NLOS_ROOMS = {"home_lab_right", "home_lab(R)", "right_home_lab"}
    _NLOS_SIGNAL_VARIANCE_THRESHOLD = 0.005

    def __init__(
        self,
        model_path: str = "",
        normalizer_dir: str = "",
        device_str: str | None = None,
    ) -> None:
        self.model_path = model_path or settings.TCN_MODEL_PATH
        self.normalizer_dir = normalizer_dir or settings.TCN_NORMALIZER_DIR
        self.device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: TemporalConvTransformer | None = None
        self.load_error: str | None = None
        self.decision_threshold = 0.5

        from app.services.csi_preprocessor import CsiZScoreNormalizer

        self._normalizer: CsiZScoreNormalizer | None = None
        try:
            self._normalizer = CsiZScoreNormalizer.load(self.normalizer_dir)
        except FileNotFoundError:
            pass

        self._load_decision_threshold()
        self._load_model()

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
            "detector_mode": "tcn",
            "model_loaded": self.model_loaded,
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": str(self.device),
            "num_classes": 2,
            "class_names": self.class_names,
            "input_shape": self.input_shape,
            "load_error": self.load_error,
            "normalizer_loaded": self.normalizer_loaded,
            "nlos_threshold": settings.NLOS_FALL_THRESHOLD,
            "default_threshold": settings.FALL_CONFIDENCE_THRESHOLD,
            "decision_threshold": self.decision_threshold,
        }

    def predict(self, csi_2d: torch.Tensor, frame: CsiFrame) -> DetectionResult:
        if self.model is None:
            raise RuntimeError(self.load_error or "TCN+Transformer model is not loaded")

        if csi_2d.dim() == 2:
            csi_2d = csi_2d.unsqueeze(0)

        if self._normalizer is not None:
            csi_2d = self._normalizer.normalize(csi_2d)

        x = csi_2d.transpose(1, 2).contiguous()
        x = x.to(self.device)
        self.model.eval()
        with torch.no_grad():
            prob_fall = float(self.model.predict_proba(x).item())

        prob_non_fall = 1.0 - prob_fall
        predicted_label = "fall" if prob_fall >= self.decision_threshold else "non_fall"
        threshold = self._get_confidence_threshold(frame.room, csi_2d)

        if predicted_label == "fall" and prob_fall >= threshold:
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
            reason=f"TCN+Transformer predicted {predicted_label} (threshold={threshold:.2f})",
            activity_score=round(prob_fall, 4),
            features={
                "model": self.model_name,
                "input_shape": self.input_shape,
                "prob_non_fall": round(prob_non_fall, 6),
                "prob_fall": round(prob_fall, 6),
                "threshold_used": threshold,
                "decision_threshold": self.decision_threshold,
                "true_label": frame.label or frame.simulated_label,
                "source": frame.source,
                "room": frame.room,
            },
        )

    def predict_from_numpy(self, data_2d: np.ndarray, frame: CsiFrame) -> DetectionResult:
        t = torch.from_numpy(data_2d.astype(np.float32))
        return self.predict(t, frame)

    def _load_model(self) -> None:
        path = Path(self.model_path)
        if not path.exists():
            self.load_error = f"TCN+Transformer model file not found: {self.model_path}"
            return

        try:
            model = TemporalConvTransformer()
            state_dict = torch.load(path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.model = model
            self.load_error = None
        except Exception as exc:
            self.model = None
            self.load_error = str(exc)

    def _load_decision_threshold(self) -> None:
        try:
            import json as _json

            base = Path(self.model_path).parent
            p = base / "training_results.json"
            if not p.exists():
                return
            d = _json.loads(p.read_text(encoding="utf-8", errors="replace"))
            thr = d.get("decision_threshold")
            if thr is None:
                return
            thr_f = float(thr)
            if 0.0 < thr_f < 1.0:
                self.decision_threshold = thr_f
        except Exception:
            return

    def _get_confidence_threshold(self, room: str, tensor: torch.Tensor | None = None) -> float:
        lower = (room or "").lower()
        if any(tag in lower for tag in ("home_lab_right", "home_lab(r)", "right")):
            return settings.NLOS_FALL_THRESHOLD

        if tensor is not None:
            try:
                var = float(tensor.float().var(dim=(0, 1)).mean().item())
                if var < self._NLOS_SIGNAL_VARIANCE_THRESHOLD:
                    return settings.NLOS_FALL_THRESHOLD
            except Exception:
                pass

        return settings.FALL_CONFIDENCE_THRESHOLD
