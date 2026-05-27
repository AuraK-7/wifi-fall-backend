import numpy as np

from app.core.config import settings
from app.schemas.csi import ActivityLabel, CsiFrame, DetectionResult, RiskLevel


class SimpleFallDetector:
    def __init__(self) -> None:
        self.recent_energy: list[float] = []
        self.recent_labels: list[str] = []

    def reset(self) -> None:
        self.recent_energy.clear()
        self.recent_labels.clear()

    def extract_features(self, frame: CsiFrame) -> dict[str, float]:
        subcarriers = np.array(frame.subcarriers, dtype=float)
        variance = float(np.var(subcarriers))
        energy = variance * 800.0

        return {
            "mean": float(np.mean(subcarriers)),
            "std": float(np.std(subcarriers)),
            "var": variance,
            "max": float(np.max(subcarriers)),
            "min": float(np.min(subcarriers)),
            "peak_to_peak": float(np.ptp(subcarriers)),
            "energy": energy,
        }

    def predict(self, frame: CsiFrame) -> DetectionResult:
        features = self.extract_features(frame)
        energy = features["energy"]
        history_mean = self._history_mean()

        predicted_label: ActivityLabel
        risk_level: RiskLevel
        confidence: float
        reason: str

        is_sudden_high_energy = energy > settings.HIGH_ENERGY_THRESHOLD and (
            history_mean == 0.0 or energy > history_mean * 2.5
        )

        if is_sudden_high_energy:
            predicted_label = "fall"
            risk_level = "high"
            confidence = min(energy / max(settings.HIGH_ENERGY_THRESHOLD, 1e-6), 1.0)
            reason = "High sudden CSI variance detected"
        elif energy >= settings.HIGH_ENERGY_THRESHOLD * 0.35:
            predicted_label = "walking"
            risk_level = "medium"
            confidence = min(energy / settings.HIGH_ENERGY_THRESHOLD, 0.74)
            reason = "Moderate movement energy"
        elif energy <= settings.LOW_ACTIVITY_THRESHOLD:
            predicted_label = "empty" if features["peak_to_peak"] < 0.08 else "lying"
            risk_level = "low"
            confidence = min(
                (settings.LOW_ACTIVITY_THRESHOLD - energy) / settings.LOW_ACTIVITY_THRESHOLD,
                1.0,
            )
            confidence = max(confidence, 0.5)
            reason = "Low activity energy"
        else:
            predicted_label = "unknown"
            risk_level = "low"
            confidence = 0.4
            reason = "CSI energy pattern is uncertain"

        confidence = max(0.0, min(confidence, 1.0))
        alert = predicted_label == "fall" and confidence >= settings.FALL_CONFIDENCE_THRESHOLD
        self._remember(energy, predicted_label)

        return DetectionResult(
            timestamp=frame.timestamp,
            room=frame.room,
            predicted_label=predicted_label,
            confidence=round(confidence, 4),
            risk_level=risk_level,
            alert=alert,
            reason=reason,
        )

    def _history_mean(self) -> float:
        if not self.recent_energy:
            return 0.0
        return float(np.mean(self.recent_energy))

    def _remember(self, energy: float, label: ActivityLabel) -> None:
        self.recent_energy.append(energy)
        self.recent_labels.append(label)

        max_history = settings.CSI_WINDOW_SIZE
        if len(self.recent_energy) > max_history:
            self.recent_energy = self.recent_energy[-max_history:]
        if len(self.recent_labels) > max_history:
            self.recent_labels = self.recent_labels[-max_history:]
