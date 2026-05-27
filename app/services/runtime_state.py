import time
from typing import Any

from app.schemas.csi import CsiFrame, DetectionResult, RecentResultItem


class RuntimeState:
    def __init__(self, max_items: int = 300) -> None:
        self.latest_frame: CsiFrame | None = None
        self.latest_result: DetectionResult | None = None
        self.recent_items: list[RecentResultItem] = []
        self.max_items = max_items
        self.alert_count = 0
        self.total_frames = 0
        self.started_at = time.time()

    def add(self, frame: CsiFrame, result: DetectionResult) -> None:
        self.latest_frame = frame
        self.latest_result = result
        self.recent_items.append(RecentResultItem(frame=frame, result=result))

        if len(self.recent_items) > self.max_items:
            self.recent_items = self.recent_items[-self.max_items :]

        self.total_frames += 1
        if result.alert:
            self.alert_count += 1

    def get_latest(self) -> dict[str, Any]:
        if self.latest_frame is None or self.latest_result is None:
            return {
                "frame": None,
                "result": None,
            }

        return {
            "frame": self.latest_frame,
            "result": self.latest_result,
        }

    def get_recent(self, limit: int = 50) -> list[RecentResultItem]:
        return list(reversed(self.recent_items[-limit:]))

    def get_summary(self) -> dict[str, Any]:
        return {
            "total_frames": self.total_frames,
            "alert_count": self.alert_count,
            "latest_label": (
                self.latest_result.predicted_label if self.latest_result is not None else None
            ),
            "latest_risk_level": (
                self.latest_result.risk_level if self.latest_result is not None else None
            ),
            "uptime_seconds": round(time.time() - self.started_at, 3),
        }
