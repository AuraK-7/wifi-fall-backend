import time

import numpy as np

from app.core.config import settings
from app.schemas.csi import ActivityLabel, CsiFrame


class CsiStreamSimulator:
    def __init__(
        self,
        room: str = settings.DEFAULT_ROOM,
        subcarrier_count: int = settings.CSI_SUBCARRIER_COUNT,
        interval_ms: int = settings.CSI_FRAME_INTERVAL_MS,
    ) -> None:
        self.room = room
        self.subcarrier_count = subcarrier_count
        self.interval_ms = interval_ms
        self.t = 0
        self.current_label: ActivityLabel = "walking"

    def set_label(self, label: ActivityLabel) -> None:
        self.current_label = label

    def next_frame(self, label: ActivityLabel | None = None) -> CsiFrame:
        active_label = label or self.current_label
        generators = {
            "empty": self._empty_signal,
            "walking": self._walking_signal,
            "sitting": self._sitting_signal,
            "lying": self._lying_signal,
            "fall": self._fall_signal,
            "unknown": self._walking_signal,
        }
        subcarriers = generators[active_label]()
        self.t += 1

        return CsiFrame(
            timestamp=time.time(),
            room=self.room,
            subcarriers=subcarriers.round(4).tolist(),
            simulated_label=active_label,
        )

    def _base_signal(self) -> np.ndarray:
        index = np.arange(self.subcarrier_count)
        spatial_curve = 1.0 + 0.08 * np.sin(index / 6.0)
        slow_drift = 0.03 * np.sin(self.t / 18.0 + index / 20.0)
        noise = np.random.normal(0.0, 0.01, self.subcarrier_count)
        return spatial_curve + slow_drift + noise

    def _empty_signal(self) -> np.ndarray:
        noise = np.random.normal(0.0, 0.006, self.subcarrier_count)
        return self._base_signal() * 0.92 + noise

    def _walking_signal(self) -> np.ndarray:
        index = np.arange(self.subcarrier_count)
        gait_wave = 0.12 * np.sin(self.t / 3.0 + index / 7.0)
        secondary_wave = 0.04 * np.sin(self.t / 1.8 + index / 13.0)
        noise = np.random.normal(0.0, 0.025, self.subcarrier_count)
        return self._base_signal() + gait_wave + secondary_wave + noise

    def _sitting_signal(self) -> np.ndarray:
        index = np.arange(self.subcarrier_count)
        transition_strength = max(0.0, 1.0 - (self.t % 40) / 12.0)
        transition = transition_strength * 0.18 * np.sin(index / 5.0 + self.t / 2.0)
        noise_scale = 0.018 if transition_strength > 0 else 0.01
        noise = np.random.normal(0.0, noise_scale, self.subcarrier_count)
        return self._base_signal() + transition + noise

    def _lying_signal(self) -> np.ndarray:
        breathing_wave = 0.035 * np.sin(self.t / 8.0)
        noise = np.random.normal(0.0, 0.012, self.subcarrier_count)
        return self._base_signal() * 0.96 + breathing_wave + noise

    def _fall_signal(self) -> np.ndarray:
        index = np.arange(self.subcarrier_count)
        phase = self.t % 50

        if phase < 8:
            impact = (1.0 - phase / 8.0) * 0.45 * np.sin(index / 3.5 + self.t)
            burst = np.random.normal(0.0, 0.08, self.subcarrier_count)
            return self._base_signal() + impact + burst

        low_activity = np.random.normal(0.0, 0.01, self.subcarrier_count)
        return self._base_signal() * 0.9 + low_activity
