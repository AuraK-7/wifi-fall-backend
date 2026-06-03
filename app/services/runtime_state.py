import time
from typing import Any

from app.schemas.csi import AnalyticsSnapshot, CsiFrame, DetectionResult, RecentResultItem


class RuntimeState:
    def __init__(self, max_items: int = 300, max_analytics: int = 600) -> None:
        self.latest_frame: CsiFrame | None = None
        self.latest_result: DetectionResult | None = None
        self.latest_analytics: AnalyticsSnapshot | None = None
        self.recent_items: list[RecentResultItem] = []
        self.max_items = max_items
        self.alert_count = 0
        self.total_frames = 0
        self.started_at = time.time()
        self._analytics_buffer: dict[int, dict[str, Any]] = {}
        self._analytics_frame_ids: list[int] = []
        self._frame_id_timestamps: dict[int, float] = {}
        self._max_analytics = max_analytics
        self._pending_chains: dict[int, list[dict[str, Any]]] = {}
        self._completed_chains: dict[int, list[dict[str, Any]]] = {}

    def start_evidence_chain(self, alert_frame_id: int) -> None:
        """Begin collecting evidence for an alert. Pre-fill BEFORE frames from buffer."""
        chain: list[dict[str, Any]] = []
        for fid in self._analytics_frame_ids:
            if alert_frame_id - 120 <= fid <= alert_frame_id:
                snap = self._analytics_buffer.get(fid)
                if not snap:
                    continue
                entry: dict[str, Any] = {"window_index": fid, "analytics": snap}
                for item in self.recent_items:
                    if item.frame.frame_id == fid:
                        entry["predicted_label"] = item.result.predicted_label
                        entry["confidence"] = item.result.confidence
                        break
                chain.append(entry)
        self._pending_chains[alert_frame_id] = chain

    def feed_pending_chains(self, frame: CsiFrame, result: DetectionResult, analytics: AnalyticsSnapshot | None) -> None:
        """Feed new frames to pending evidence chains. Finalize when 40 after-frames collected."""
        if not self._pending_chains:
            return
        fid = frame.frame_id
        entry: dict[str, Any] = {
            "window_index": fid,
            "analytics": analytics.model_dump() if analytics else None,
            "predicted_label": result.predicted_label,
            "confidence": result.confidence,
        }
        for alert_fid, chain in list(self._pending_chains.items()):
            after_count = fid - alert_fid
            if 0 < after_count <= 40:
                chain.append(entry)
            if after_count >= 40:
                self._completed_chains[alert_fid] = chain
                del self._pending_chains[alert_fid]

    def store_evidence_chain(self, alert_frame_id: int, chain: list[dict[str, Any]]) -> None:
        """Directly store a pre-built evidence chain (used by single-shot demo trigger)."""
        self._completed_chains[alert_frame_id] = chain

    def get_evidence_chain(self, alert_frame_id: int) -> list[dict[str, Any]] | None:
        """Return completed evidence chain, or pending chain if still collecting."""
        if alert_frame_id in self._completed_chains:
            return self._completed_chains[alert_frame_id]
        if alert_frame_id in self._pending_chains:
            return self._pending_chains[alert_frame_id]
        return None


    def add(
        self,
        frame: CsiFrame,
        result: DetectionResult,
        analytics: AnalyticsSnapshot | None = None,
    ) -> None:
        self.latest_frame = frame
        self.latest_result = result
        self.latest_analytics = analytics
        self.recent_items.append(RecentResultItem(frame=frame, result=result))

        if len(self.recent_items) > self.max_items:
            self.recent_items = self.recent_items[-self.max_items:]

        self.total_frames += 1
        if result.alert:
            self.alert_count += 1

        if analytics is not None:
            fid = frame.frame_id
            self._analytics_buffer[fid] = analytics.model_dump()
            self._analytics_frame_ids.append(fid)
            self._frame_id_timestamps[fid] = frame.timestamp
            if len(self._analytics_frame_ids) > self._max_analytics:
                oldest = self._analytics_frame_ids.pop(0)
                self._analytics_buffer.pop(oldest, None)
                self._frame_id_timestamps.pop(oldest, None)

        self.feed_pending_chains(frame, result, analytics)

    def find_closest_frame_id(self, timestamp: float) -> int:
        best_fid = 0
        best_dist = float("inf")
        for fid, ts in self._frame_id_timestamps.items():
            dist = abs(ts - timestamp)
            if dist < best_dist:
                best_dist = dist
                best_fid = fid
        return best_fid

    def get_analytics_by_frame_id(self, frame_id: int) -> dict[str, Any] | None:
        return self._analytics_buffer.get(frame_id)

    def get_analytics_window(
        self, centre_frame_id: int, before: int = 100, after: int = 100
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for fid in self._analytics_frame_ids:
            if centre_frame_id - before <= fid <= centre_frame_id + after:
                snap = self._analytics_buffer.get(fid)
                if snap is not None:
                    results.append({"frame_id": fid, **snap})
        return results

    def get_latest(self) -> dict[str, Any]:
        if self.latest_frame is None or self.latest_result is None:
            return {"frame": None, "result": None, "analytics": None}
        return {"frame": self.latest_frame, "result": self.latest_result, "analytics": self.latest_analytics}

    def get_recent(self, limit: int = 50) -> list[RecentResultItem]:
        return list(reversed(self.recent_items[-limit:]))

    def get_summary(self) -> dict[str, Any]:
        return {
            "total_frames": self.total_frames,
            "alert_count": self.alert_count,
            "latest_label": self.latest_result.predicted_label if self.latest_result is not None else None,
            "latest_risk_level": self.latest_result.risk_level if self.latest_result is not None else None,
            "uptime_seconds": round(time.time() - self.started_at, 3),
        }
