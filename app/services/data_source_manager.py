from typing import Any

from app.core.config import settings
from app.data_sources.base import BaseCsiSource
from app.data_sources.csv_replay_source import CsvReplayCsiSource
from app.data_sources.enetfall_mat_source import (
    DEFAULT_ENETFALL_DATASETS,
    EnetFallMatDataSource,
)
from app.schemas.csi import ActivityLabel


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
            self.current_source = CsvReplayCsiSource(
                csv_path="data/wifi_csi_har_dataset/room_1/1/data.csv",
                room="room_1",
                device_id="csv-node-001",
                label="unknown",
            )
            self.source_mode = "csv"
            self.load_error = str(exc)

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
