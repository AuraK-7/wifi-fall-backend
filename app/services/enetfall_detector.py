from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchvision

from app.core.config import settings
from app.schemas.csi import CsiFrame, DetectionResult


class ENetFallDetector:
    model_name = "efficientnet_b0_enetfall"
    class_names = ["non_fall", "fall"]
    input_shape = [3, 625, 30]

    def __init__(self, model_path: str = settings.ENETFALL_MODEL_PATH) -> None:
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: nn.Module | None = None
        self.load_error: str | None = None
        self._load_model()

    @property
    def model_loaded(self) -> bool:
        return self.model is not None and self.load_error is None

    def reset(self) -> None:
        return None

    def get_status(self) -> dict[str, Any]:
        return {
            "detector_mode": "enetfall",
            "model_loaded": self.model_loaded,
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": str(self.device),
            "num_classes": 2,
            "class_names": self.class_names,
            "input_shape": self.input_shape,
            "load_error": self.load_error,
        }

    def predict_window(self, frame: CsiFrame, input_tensor: torch.Tensor) -> DetectionResult:
        if self.model is None:
            raise RuntimeError(self.load_error or "ENetFall model is not loaded")

        tensor = input_tensor.to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            pred_idx = int(torch.argmax(probs, dim=1).item())
            confidence = float(probs[0][pred_idx].item())

        predicted_label = "fall" if pred_idx == 1 else "non_fall"
        prob_non_fall = float(probs[0][0].item())
        prob_fall = float(probs[0][1].item())

        if predicted_label == "fall" and confidence >= 0.70:
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
            confidence=round(confidence, 4),
            risk_level=risk_level,
            alert=alert,
            reason=f"ENetFall EfficientNet-B0 model predicted {predicted_label}",
            activity_score=round(prob_fall, 4),
            features={
                "model": self.model_name,
                "input_shape": self.input_shape,
                "prob_non_fall": round(prob_non_fall, 6),
                "prob_fall": round(prob_fall, 6),
                "true_label": frame.label or frame.simulated_label,
                "source": frame.source,
                "window_shape": frame.window_shape,
            },
        )

    def _load_model(self) -> None:
        path = Path(self.model_path)
        if not path.exists():
            self.load_error = f"ENetFall model file not found: {self.model_path}"
            return

        try:
            try:
                model = torchvision.models.efficientnet_b0(pretrained=True)
            except Exception:
                model = torchvision.models.efficientnet_b0(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.SiLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.SiLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 2),
            )
            state_dict = torch.load(path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.model = model
            self.load_error = None
        except Exception as exc:  # pragma: no cover - depends on local model/runtime
            self.model = None
            self.load_error = str(exc)
