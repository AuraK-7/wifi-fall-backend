from typing import Any

from app.data_sources.base import BaseCsiSource
from app.data_sources.csv_replay_source import CsvReplayCsiSource
from app.schemas.csi import ActivityLabel

DEFAULT_CSV_PATH = "data/wifi_csi_har_dataset/room_1/1/data.csv"


class DataSourceManager:
    def __init__(
        self,
        csv_path: str = DEFAULT_CSV_PATH,
        room: str = "room_1",
        device_id: str = "csv-node-001",
        label: ActivityLabel = "unknown",
    ) -> None:
        self.source_mode = "csv"
        self.current_source: BaseCsiSource = CsvReplayCsiSource(
            csv_path=csv_path,
            room=room,
            device_id=device_id,
            label=label,
        )

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
        return self.current_source

    def get_current_source(self) -> BaseCsiSource:
        return self.current_source

    def get_status(self) -> dict[str, Any]:
        return {
            "source_mode": self.source_mode,
            "current_source": self.current_source.get_status(),
        }
