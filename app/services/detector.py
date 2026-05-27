import numpy as np

from app.core.config import settings
from app.schemas.csi import CsiFrame, DetectionResult


class FallDetector:
    def detect(self, frame: CsiFrame) -> DetectionResult:
        subcarriers = np.array(frame.subcarriers, dtype=float)
        energy = float(np.sum(np.square(subcarriers)))
        confidence = min(energy / settings.HIGH_ENERGY_THRESHOLD, 1.0)
        predicted_label = "fall" if confidence >= settings.FALL_CONFIDENCE_THRESHOLD else "unknown"
        risk_level = "high" if predicted_label == "fall" else "low"
        alert = predicted_label == "fall"

        return DetectionResult(
            timestamp=frame.timestamp,
            room=frame.room,
            predicted_label=predicted_label,
            confidence=round(confidence, 4),
            risk_level=risk_level,
            alert=alert,
            reason="High CSI energy detected" if alert else "No fall pattern detected",
        )
