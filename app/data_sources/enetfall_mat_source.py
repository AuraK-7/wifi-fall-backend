import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import scipy.io as sio
import torch
from torchvision import transforms as T

from app.core.config import settings
from app.data_sources.base import BaseCsiSource
from app.schemas.csi import ActivityLabel, CsiFrame

DEFAULT_ENETFALL_DATASETS = [
    "dataset_home_lab(L).mat",
    "dataset_home_lab(R).mat",
    "dataset_lecture_room.mat",
    "dataset_living_room.mat",
    "dataset_meeting_room.mat",
]

DATASET_ROOM_MAP = {
    "dataset_home_lab(L).mat": "home_lab_left",
    "dataset_home_lab(R).mat": "home_lab_right",
    "dataset_lecture_room.mat": "lecture_room",
    "dataset_living_room.mat": "living_room",
    "dataset_meeting_room.mat": "meeting_room",
}


class EnetFallMatDataSource(BaseCsiSource):
    def __init__(
        self,
        data_dir: str = settings.ENETFALL_DATA_DIR,
        dataset_names: list[str] | None = None,
        device_id: str = "enetfall-node-001",
        room: str = "home",
    ) -> None:
        self.data_dir = str(Path(data_dir))
        self.dataset_names = dataset_names or DEFAULT_ENETFALL_DATASETS
        self.device_id = device_id
        self.room = room
        self.current_label: ActivityLabel = "unknown"
        self.frame_id = 0
        self.current_index = 0
        self.loop = True

        data_all, labels_all, sample_rooms = self._load_datasets()
        self.labels = labels_all.astype(np.int64)
        self.sample_rooms = sample_rooms
        self.processed_tensor = self._preprocess(data_all)

        if self.processed_tensor.shape[0] == 0:
            raise ValueError("ENetFall dataset is empty")

    def next_frame(self) -> CsiFrame:
        if self.current_index >= self.total_samples:
            self.current_index = 0

        sample = self.processed_tensor[self.current_index]
        label = self._label_from_idx(int(self.labels[self.current_index]))
        sample_room = self.sample_rooms[self.current_index] or self.room
        self.current_index += 1
        self.frame_id += 1

        return CsiFrame(
            frame_id=self.frame_id,
            device_id=self.device_id,
            timestamp=time.time(),
            room=sample_room,
            subcarriers=self._preview_subcarriers(sample),
            simulated_label=label,
            source="enetfall_mat",
            window_shape=list(sample.shape),
            label=label,
        )

    def next_window(self) -> tuple[CsiFrame, torch.Tensor, ActivityLabel]:
        frame = self.next_frame()
        index = (self.current_index - 1) % self.total_samples
        window = self.processed_tensor[index].unsqueeze(0)
        label = frame.label or frame.simulated_label
        return frame, window, label

    def get_window_at(self, index: int) -> torch.Tensor | None:
        """Return [1,3,625,30] tensor at *index* without advancing the stream pointer."""
        if index < 0 or index >= self.total_samples:
            return None
        return self.processed_tensor[index].unsqueeze(0).clone()

    def get_label_at(self, index: int) -> ActivityLabel:
        return self._label_from_idx(int(self.labels[index]))

    def get_room_at(self, index: int) -> str:
        return self.sample_rooms[index] or self.room

    @staticmethod
    def create_reader(
        data_dir: str | None = None,
        dataset_names: list[str] | None = None,
    ) -> "EnetFallMatDataSource":
        """Create an independent reader instance (does not affect WebSocket stream)."""
        return EnetFallMatDataSource(
            data_dir=data_dir or settings.ENETFALL_DATA_DIR,
            dataset_names=dataset_names or DEFAULT_ENETFALL_DATASETS,
            device_id="enetfall-reader",
            room="archive",
        )

    def set_label(self, label: ActivityLabel) -> None:
        self.current_label = label

    def set_room(self, room: str) -> None:
        self.room = room

    def set_device(self, device_id: str) -> None:
        self.device_id = device_id

    def get_status(self) -> dict[str, Any]:
        return {
            "type": "enetfall_mat",
            "data_dir": self.data_dir,
            "dataset_names": self.dataset_names,
            "total_samples": self.total_samples,
            "current_index": self.current_index,
            "device_id": self.device_id,
            "room": self.room,
            "loop": self.loop,
            "window_shape": [3, 625, 30],
        }

    @property
    def total_samples(self) -> int:
        return int(self.processed_tensor.shape[0])

    def _load_datasets(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        data_parts: list[np.ndarray] = []
        label_parts: list[np.ndarray] = []
        sample_rooms: list[str] = []

        for dataset_name in self.dataset_names:
            path = Path(self.data_dir) / dataset_name
            if not path.exists():
                raise FileNotFoundError(f"ENetFall dataset not found: {path}")

            mat = sio.loadmat(path)
            if "dataset_CSI_t" not in mat or "dataset_labels" not in mat:
                raise ValueError(f"Invalid ENetFall MAT file: {path}")

            data = np.asarray(mat["dataset_CSI_t"])
            labels = np.asarray(mat["dataset_labels"]).reshape(-1)
            if data.ndim != 3 or data.shape[1:] != (625, 90):
                raise ValueError(f"Unexpected ENetFall data shape in {path}: {data.shape}")
            if data.shape[0] != labels.shape[0]:
                raise ValueError(f"Data/label count mismatch in {path}")

            data_parts.append(data)
            label_parts.append(labels)
            sample_rooms.extend([DATASET_ROOM_MAP.get(dataset_name, self.room)] * data.shape[0])

        return (
            np.concatenate(data_parts, axis=0),
            np.concatenate(label_parts, axis=0),
            sample_rooms,
        )

    def _preprocess(self, data_all: np.ndarray) -> torch.Tensor:
        num_instances = data_all.shape[0]
        data_all_3ch = np.ndarray(shape=(num_instances, 3, 625, 30), dtype=np.float32)
        data_all_3ch[:, 0, :, :] = data_all[:, :, 0:90:3]
        data_all_3ch[:, 1, :, :] = data_all[:, :, 1:90:3]
        data_all_3ch[:, 2, :, :] = data_all[:, :, 2:90:3]

        data_tensor = torch.from_numpy(data_all_3ch).type(torch.FloatTensor).view(
            num_instances,
            3,
            625,
            30,
        )
        transform = T.Compose(
            [
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        for idx in range(num_instances):
            data_tensor[idx] = transform(data_tensor[idx])

        max_value = torch.max(data_tensor)
        if float(max_value) != 0.0:
            data_tensor = data_tensor / max_value

        mean_value = torch.mean(data_tensor)
        return data_tensor - mean_value

    def _preview_subcarriers(self, sample: torch.Tensor) -> list[float]:
        avg_across_antennas = sample.mean(dim=0)          # [625, 30]
        latest_slice = avg_across_antennas[-1, :]          # [30]
        return [round(float(v), 6) for v in latest_slice]

    def _label_from_idx(self, label_idx: int) -> ActivityLabel:
        if label_idx == 1:
            return "fall"
        if label_idx == 0:
            return "non_fall"
        return cast(ActivityLabel, self.current_label)
