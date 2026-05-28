import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from app.core.config import settings
from app.data_sources.base import BaseCsiSource
from app.schemas.csi import ActivityLabel, CsiFrame

VALID_LABELS: set[str] = {"empty", "walking", "sitting", "lying", "fall", "unknown"}
METADATA_COLUMNS: set[str] = {
    "timestamp",
    "time",
    "room",
    "device_id",
    "device",
    "label",
    "activity",
    "activity_label",
    "simulated_label",
}


class CsvReplayCsiSource(BaseCsiSource):
    def __init__(
        self,
        csv_path: str,
        room: str = "real_room",
        device_id: str = "csv-node-001",
        label: ActivityLabel = "unknown",
    ) -> None:
        self.csv_path = str(Path(csv_path))
        self.room = room
        self.device_id = device_id
        self.current_label = label
        self.subcarrier_count = settings.CSI_SUBCARRIER_COUNT
        self.frame_id = 0
        self.row_index = 0
        self.data = self._load_csv(self.csv_path)
        self.numeric_columns = self._detect_numeric_columns()

        if self.data.empty:
            raise ValueError("CSV file is empty")
        if not self.numeric_columns:
            raise ValueError("CSV file does not contain numeric CSI columns")

    def next_frame(self) -> CsiFrame:
        if self.row_index >= len(self.data):
            self.row_index = 0

        row = self.data.iloc[self.row_index]
        self.row_index += 1
        self.frame_id += 1

        label = self._label_from_row(row)
        room = self._string_from_row(row, "room", self.room)
        device_id = self._string_from_row(row, "device_id", self.device_id)
        timestamp = self._timestamp_from_row(row)
        subcarriers = self._fixed_length_subcarriers(row)

        return CsiFrame(
            frame_id=self.frame_id,
            device_id=device_id,
            timestamp=timestamp,
            room=room,
            subcarriers=subcarriers,
            simulated_label=label,
        )

    def set_label(self, label: ActivityLabel) -> None:
        self.current_label = label

    def set_room(self, room: str) -> None:
        self.room = room

    def set_device(self, device_id: str) -> None:
        self.device_id = device_id

    def get_status(self) -> dict[str, Any]:
        return {
            "type": "csv",
            "csv_path": self.csv_path,
            "current_label": self.current_label,
            "room": self.room,
            "device_id": self.device_id,
            "subcarrier_count": self.subcarrier_count,
            "total_rows": len(self.data),
            "row_index": self.row_index,
            "loop": True,
        }

    def _load_csv(self, csv_path: str) -> pd.DataFrame:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        return pd.read_csv(path)

    def _detect_numeric_columns(self) -> list[Any]:
        numeric_data = self.data.apply(pd.to_numeric, errors="coerce")
        columns: list[Any] = []
        for column in self.data.columns:
            normalized_name = str(column).strip().lower()
            if normalized_name in METADATA_COLUMNS:
                continue
            if numeric_data[column].notna().any():
                columns.append(column)
        return columns

    def _fixed_length_subcarriers(self, row: pd.Series) -> list[float]:
        values = pd.to_numeric(row[self.numeric_columns], errors="coerce")
        array = values.dropna().to_numpy(dtype=float)
        array = self._drop_monotonic_index_prefix(array)

        if len(array) == 0:
            array = np.zeros(self.subcarrier_count, dtype=float)
        elif len(array) < self.subcarrier_count:
            array = np.pad(array, (0, self.subcarrier_count - len(array)), mode="edge")
        elif len(array) > self.subcarrier_count:
            array = array[: self.subcarrier_count]

        return np.round(array.astype(float), 4).tolist()

    def _drop_monotonic_index_prefix(self, array: np.ndarray) -> np.ndarray:
        max_prefix = min(len(array), 256)
        for prefix_len in range(max_prefix, 7, -1):
            if np.allclose(array[:prefix_len], np.arange(prefix_len), atol=1e-9):
                return array[prefix_len:]
        return array

    def _label_from_row(self, row: pd.Series) -> ActivityLabel:
        for column in ("label", "activity", "activity_label", "simulated_label"):
            if column in row.index and pd.notna(row[column]):
                label = self._normalize_label(str(row[column]))
                if label is not None:
                    return label
        return self.current_label

    def _normalize_label(self, value: str) -> ActivityLabel | None:
        normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
        mapping = {
            "walk": "walking",
            "walking": "walking",
            "sit": "sitting",
            "sitting": "sitting",
            "lie": "lying",
            "lying": "lying",
            "fall": "fall",
            "falling": "fall",
            "empty": "empty",
            "none": "empty",
            "unknown": "unknown",
        }
        mapped = mapping.get(normalized, normalized)
        if mapped in VALID_LABELS:
            return cast(ActivityLabel, mapped)
        return None

    def _string_from_row(self, row: pd.Series, column: str, default: str) -> str:
        if column in row.index and pd.notna(row[column]):
            value = str(row[column]).strip()
            if value:
                return value
        return default

    def _timestamp_from_row(self, row: pd.Series) -> float:
        for column in ("timestamp", "time"):
            if column in row.index and pd.notna(row[column]):
                try:
                    return float(row[column])
                except (TypeError, ValueError):
                    break
        return time.time()
