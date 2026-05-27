import numpy as np

from app.core.config import settings
from app.schemas.csi import CsiFrame, FallDetectionResult


class FallDetector:
    def detect(self, frame: CsiFrame) -> FallDetectionResult:
        amplitudes = np.array(frame.amplitudes, dtype=float)
        volatility = float(np.std(amplitudes))
        score = min(volatility / settings.FALL_THRESHOLD, 1.0)
        status = "fall_suspected" if score >= 0.8 else "normal"

        return FallDetectionResult(
            timestamp=frame.timestamp,
            device_id=frame.device_id,
            score=round(score, 4),
            status=status,
            message="Fall suspected" if status == "fall_suspected" else "Normal activity",
        )
