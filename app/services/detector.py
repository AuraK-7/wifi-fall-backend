import numpy as np

from app.core.config import settings
from app.schemas.csi import ActivityLabel, CsiFrame, DetectionResult, RiskLevel


class SimpleFallDetector:
    def __init__(self, max_history: int = 200) -> None:
        self.recent_energy: list[float] = []
        self.recent_results: list[DetectionResult] = []
        self.max_history = max_history
        self.fall_candidate_frames = 0
        self.low_activity_after_fall_frames = 0
        self._last_mean: float | None = None

    def reset(self) -> None:
        self.recent_energy.clear()
        self.recent_results.clear()
        self.fall_candidate_frames = 0
        self.low_activity_after_fall_frames = 0
        self._last_mean = None

    def extract_features(self, frame: CsiFrame) -> dict[str, float]:
        subcarriers = np.array(frame.subcarriers, dtype=float)
        mean_value = float(np.mean(subcarriers))
        variance = float(np.var(subcarriers))
        energy = variance * 800.0
        diff_energy = 0.0 if self._last_mean is None else abs(mean_value - self._last_mean)
        self._last_mean = mean_value

        return {
            "mean": mean_value,
            "std": float(np.std(subcarriers)),
            "var": variance,
            "max": float(np.max(subcarriers)),
            "min": float(np.min(subcarriers)),
            "peak_to_peak": float(np.ptp(subcarriers)),
            "energy": energy,
            "diff_energy": diff_energy,
        }

    def compute_activity_score(self, features: dict[str, float]) -> float:
        energy_score = features["energy"] / settings.HIGH_ENERGY_THRESHOLD
        diff_score = features["diff_energy"] / 0.2
        score = 0.75 * energy_score + 0.25 * diff_score
        return max(0.0, min(1.0, score))

    def predict(self, frame: CsiFrame) -> DetectionResult:
        features = self.extract_features(frame)
        energy = features["energy"]
        peak_to_peak = features["peak_to_peak"]
        activity_score = self.compute_activity_score(features)
        window_mean = self._window_energy_mean()

        predicted_label: ActivityLabel = "unknown"
        risk_level: RiskLevel = "low"
        confidence = 0.4
        reason = "CSI energy pattern is uncertain"

        sudden_energy = energy > settings.HIGH_ENERGY_THRESHOLD and (
            window_mean == 0.0 or energy > window_mean * 2.2
        )
        large_peak = peak_to_peak >= 0.65
        fall_candidate = sudden_energy and large_peak

        if fall_candidate:
            self.fall_candidate_frames = max(self.fall_candidate_frames, 3)
            self.low_activity_after_fall_frames = 0
            predicted_label = "fall"
            risk_level = "high"
            confidence = min(1.0, 0.75 + activity_score * 0.25)
            reason = "High sudden CSI variance detected"
        elif self.fall_candidate_frames > 0 and activity_score <= 0.15:
            self.fall_candidate_frames -= 1
            self.low_activity_after_fall_frames += 1
            if self.low_activity_after_fall_frames >= 2:
                predicted_label = "fall"
                risk_level = "high"
                confidence = 0.9
                reason = "Low activity after fall candidate"
            else:
                predicted_label = "lying"
                risk_level = "medium"
                confidence = 0.65
                reason = "Low activity after sudden movement"
        elif activity_score >= 0.25:
            self.fall_candidate_frames = max(0, self.fall_candidate_frames - 1)
            self.low_activity_after_fall_frames = 0
            predicted_label = "walking"
            risk_level = "medium"
            confidence = min(0.74, 0.45 + activity_score * 0.4)
            reason = "Moderate movement energy"
        elif activity_score <= 0.08:
            self.fall_candidate_frames = max(0, self.fall_candidate_frames - 1)
            self.low_activity_after_fall_frames = 0
            predicted_label = self._low_activity_label(frame, peak_to_peak)
            risk_level = "low"
            confidence = max(0.5, 1.0 - activity_score * 5.0)
            reason = "Low activity energy"
        else:
            self.fall_candidate_frames = max(0, self.fall_candidate_frames - 1)
            self.low_activity_after_fall_frames = 0

        confidence = max(0.0, min(1.0, confidence))
        alert = predicted_label == "fall" and risk_level == "high" and (
            confidence >= settings.FALL_CONFIDENCE_THRESHOLD
        )

        result = DetectionResult(
            timestamp=frame.timestamp,
            room=frame.room,
            predicted_label=predicted_label,
            confidence=round(confidence, 4),
            risk_level=risk_level,
            alert=alert,
            reason=reason,
            activity_score=round(activity_score, 4),
            features={key: round(value, 6) for key, value in features.items()},
        )
        self._remember(energy, result)
        return result

    def _window_energy_mean(self) -> float:
        if not self.recent_energy:
            return 0.0
        window = self.recent_energy[-settings.CSI_WINDOW_SIZE :]
        return float(np.mean(window))

    def _low_activity_label(self, frame: CsiFrame, peak_to_peak: float) -> ActivityLabel:
        if frame.simulated_label in {"empty", "lying"}:
            return frame.simulated_label
        return "empty" if peak_to_peak < 0.08 else "lying"

    def _remember(self, energy: float, result: DetectionResult) -> None:
        self.recent_energy.append(energy)
        self.recent_results.append(result)

        if len(self.recent_energy) > self.max_history:
            self.recent_energy = self.recent_energy[-self.max_history :]
        if len(self.recent_results) > self.max_history:
            self.recent_results = self.recent_results[-self.max_history :]
