import time
from typing import Any, cast

import numpy as np

from app.core.config import settings
from app.schemas.csi import ActivityLabel, CsiFrame


class CsiStreamSimulator:
    def __init__(
        self,
        device_id: str = "sim-node-001",
        room: str = settings.DEFAULT_ROOM,
        subcarrier_count: int = settings.CSI_SUBCARRIER_COUNT,
        interval_ms: int = settings.CSI_FRAME_INTERVAL_MS,
    ) -> None:
        self.device_id = device_id
        self.room = room
        self.subcarrier_count = subcarrier_count
        self.interval_ms = interval_ms
        self.t = 0
        self.current_label: ActivityLabel = "walking"
        self.sequence: list[dict[str, Any]] = []
        self.sequence_index = 0
        self.sequence_frame_index = 0
        self.sequence_loop = True

    def set_label(self, label: ActivityLabel) -> None:
        self.current_label = label

    def set_room(self, room: str) -> None:
        self.room = room

    def set_device(self, device_id: str) -> None:
        self.device_id = device_id

    def load_sequence(self, sequence: list[dict]) -> None:
        if not sequence:
            self.clear_sequence()
            return

        normalized: list[dict[str, Any]] = []
        for item in sequence:
            label = item.get("label")
            duration_frames = item.get("duration_frames")
            if label not in {"empty", "walking", "sitting", "lying", "fall", "unknown"}:
                raise ValueError(f"Invalid sequence label: {label}")
            if not isinstance(duration_frames, int) or duration_frames <= 0:
                raise ValueError("duration_frames must be a positive integer")

            normalized.append(
                {
                    "label": cast(ActivityLabel, label),
                    "duration_frames": duration_frames,
                }
            )

        self.sequence = normalized
        self.sequence_index = 0
        self.sequence_frame_index = 0

    def clear_sequence(self) -> None:
        self.sequence = []
        self.sequence_index = 0
        self.sequence_frame_index = 0

    def _get_label_from_sequence(self) -> ActivityLabel | None:
        if not self.sequence:
            return None

        current_step = self.sequence[self.sequence_index]
        label = cast(ActivityLabel, current_step["label"])
        duration_frames = int(current_step["duration_frames"])

        self.sequence_frame_index += 1
        if self.sequence_frame_index >= duration_frames:
            self.sequence_frame_index = 0
            self.sequence_index += 1
            if self.sequence_index >= len(self.sequence):
                self.sequence_index = 0 if self.sequence_loop else len(self.sequence) - 1

        return label

    def next_frame(self, label: ActivityLabel | None = None) -> CsiFrame:
        active_label = label or self._get_label_from_sequence() or self.current_label
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
            frame_id=self.t,
            device_id=self.device_id,
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
        noise = np.random.normal(0.0, 0.003, self.subcarrier_count)
        return self._base_signal() * 0.9 + noise

    def _walking_signal(self) -> np.ndarray:
        index = np.arange(self.subcarrier_count)
        slow_wave = 0.08 * np.sin(self.t / 9.0)
        gait_wave = 0.14 * np.sin(self.t / 3.2 + index / 7.0)
        secondary_wave = 0.05 * np.sin(self.t / 1.9 + index / 13.0)
        noise = np.random.normal(0.0, 0.022, self.subcarrier_count)
        return self._base_signal() + slow_wave + gait_wave + secondary_wave + noise

    def _sitting_signal(self) -> np.ndarray:
        index = np.arange(self.subcarrier_count)
        phase = self.t % 50
        transition_strength = max(0.0, 1.0 - phase / 10.0)
        transition = transition_strength * 0.2 * np.sin(index / 5.0 + self.t / 2.0)
        noise_scale = 0.018 if transition_strength > 0 else 0.01
        noise = np.random.normal(0.0, noise_scale, self.subcarrier_count)
        return self._base_signal() + transition + noise

    def _lying_signal(self) -> np.ndarray:
        breathing_wave = 0.025 * np.sin(self.t / 8.0)
        noise = np.random.normal(0.0, 0.008, self.subcarrier_count)
        return self._base_signal() * 0.94 + breathing_wave + noise

    def _fall_signal(self) -> np.ndarray:
        index = np.arange(self.subcarrier_count)
        phase = self.t % 70

        if phase < 15:
            impact = (1.0 - phase / 15.0) * 0.55 * np.sin(index / 3.5 + self.t)
            burst = np.random.normal(0.0, 0.09, self.subcarrier_count)
            return self._base_signal() + impact + burst

        low_activity = np.random.normal(0.0, 0.008, self.subcarrier_count)
        return self._base_signal() * 0.91 + low_activity
