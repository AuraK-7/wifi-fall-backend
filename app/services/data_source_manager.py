import time
from typing import Any

import numpy as np

from app.core.config import settings
from app.data_sources.base import BaseCsiSource
from app.data_sources.csv_replay_source import CsvReplayCsiSource
from app.data_sources.enetfall_mat_source import (
    DEFAULT_ENETFALL_DATASETS,
    EnetFallMatDataSource,
)
from app.schemas.csi import ActivityLabel, CsiFrame


class SyntheticCsiSource(BaseCsiSource):
    def __init__(
        self,
        room: str = "synthetic_room",
        device_id: str = "synthetic-node-001",
        label: ActivityLabel = "unknown",
        subcarrier_count: int = settings.CSI_SUBCARRIER_COUNT,
    ) -> None:
        self.room = room
        self.device_id = device_id
        self.current_label = label
        self.subcarrier_count = subcarrier_count
        self.frame_id = 0

    def next_frame(self) -> CsiFrame:
        self.frame_id += 1
        subcarriers = np.random.normal(loc=0.0, scale=0.15, size=self.subcarrier_count)
        return CsiFrame(
            frame_id=self.frame_id,
            device_id=self.device_id,
            timestamp=time.time(),
            room=self.room,
            subcarriers=np.round(subcarriers.astype(float), 4).tolist(),
            simulated_label=self.current_label,
            source="synthetic",
            label=self.current_label,
        )

    def set_label(self, label: ActivityLabel) -> None:
        self.current_label = label

    def set_room(self, room: str) -> None:
        self.room = room

    def set_device(self, device_id: str) -> None:
        self.device_id = device_id

    def get_status(self) -> dict[str, Any]:
        return {
            "type": "synthetic",
            "room": self.room,
            "device_id": self.device_id,
            "current_label": self.current_label,
            "subcarrier_count": self.subcarrier_count,
        }


class DataSourceManager:
    def __init__(self) -> None:
        try:
            self.current_source: BaseCsiSource = EnetFallMatDataSource(
                data_dir=settings.ENETFALL_DATA_DIR,
                dataset_names=DEFAULT_ENETFALL_DATASETS,
            )
            self.source_mode = "enetfall"
            self.load_error: str | None = None
        except Exception as exc:
            try:
                self.current_source = CsvReplayCsiSource(
                    csv_path="data/wifi_csi_har_dataset/room_1/1/data.csv",
                    room="room_1",
                    device_id="csv-node-001",
                    label="unknown",
                )
                self.source_mode = "csv"
                self.load_error = str(exc)
            except Exception as csv_exc:
                self.current_source = SyntheticCsiSource(
                    room=settings.DEFAULT_ROOM,
                    device_id="synthetic-node-001",
                    label="unknown",
                )
                self.source_mode = "synthetic"
                self.load_error = f"ENetFall load failed: {exc}; CSV load failed: {csv_exc}"

    def switch_to_csv(
        self,
        csv_path: str,
        room: str = "real_room",
        device_id: str = "csv-node-001",
        label: ActivityLabel = "unknown",
    ) -> BaseCsiSource:
        self.current_source = CsvReplayCsiSource(
            csv_path=csv_path,
            room=room,
            device_id=device_id,
            label=label,
        )
        self.source_mode = "csv"
        self.load_error = None
        return self.current_source

    def switch_to_enetfall(
        self,
        data_dir: str | None = None,
        dataset_names: list[str] | None = None,
        device_id: str = "enetfall-node-001",
        room: str = "home",
    ) -> BaseCsiSource:
        self.current_source = EnetFallMatDataSource(
            data_dir=data_dir or settings.ENETFALL_DATA_DIR,
            dataset_names=dataset_names or DEFAULT_ENETFALL_DATASETS,
            device_id=device_id,
            room=room,
        )
        self.source_mode = "enetfall"
        self.load_error = None
        return self.current_source

    def get_current_source(self) -> BaseCsiSource:
        return self.current_source

    def get_status(self) -> dict[str, Any]:
        return {
            "source_mode": self.source_mode,
            "current_source": self.current_source.get_status(),
            "load_error": self.load_error,
        }
