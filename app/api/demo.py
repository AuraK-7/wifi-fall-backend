"""Single-shot ENetFall demo trigger.

Loads one window, runs EfficientNet‑B0, computes analytics for a short
evidence chain, persists the alert, stores the chain so that
``GET /api/events/{event_id}/replay`` returns usable 3D replay data,
and broadcasts the envelope to all WebSocket clients.
"""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.config import settings
from app.data_sources.enetfall_mat_source import DEFAULT_ENETFALL_DATASETS, EnetFallMatDataSource
from app.schemas.alert import AlertEventCreate
from app.schemas.csi import AnalyticsSnapshot, CsiFrame
from app.services.alert import AlertService
from app.services.connection_manager import ConnectionManager
from app.services.enetfall_detector import ENetFallDetector
from app.services.runtime_state import RuntimeState
from app.services.signal_processor import compute_analytics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/demo", tags=["demo"])

# ── Scene presets ───────────────────────────────────────────────────
SCENES: dict[str, list[str]] = {
    "living_room":   ["dataset_living_room.mat"],
    "home_lab":      ["dataset_home_lab(L).mat", "dataset_home_lab(R).mat"],
    "meeting_room":  ["dataset_meeting_room.mat"],
    "lecture_room":  ["dataset_lecture_room.mat"],
    "all":           DEFAULT_ENETFALL_DATASETS,
}

# ── Request ─────────────────────────────────────────────────────────

class DemoTriggerRequest(BaseModel):
    scene: str = Field(default="living_room")
    sample_index: int = Field(default=0, ge=0)
    room: str = Field(default="demo")
    device_id: str = Field(default="demo-node")
    evidence_before: int = Field(default=40, ge=0, le=120)
    evidence_after: int = Field(default=20, ge=0, le=80)


# ── Helpers (module-level, no closure needed) ──────────────────────

def _avatar(frame: CsiFrame, result: Any) -> dict[str, Any]:
    ds = "fallen" if (frame.label or frame.simulated_label) == "fall" else "standing"
    ps = "fallen" if result.predicted_label == "fall" else "standing"
    return {
        "display_state": ds, "dataset_state": ds, "predicted_state": ps,
        "source": "dataset_label",
        "dataset_label": frame.label or frame.simulated_label,
        "predicted_label": result.predicted_label,
        "confidence": result.confidence,
        "risk_level": result.risk_level, "alert": result.alert,
    }


# ── Router factory ──────────────────────────────────────────────────

def create_demo_router(
    detector: ENetFallDetector,
    runtime: RuntimeState,
    ws_manager: ConnectionManager,
    alert_service: AlertService,
    alert_cooldown_seconds: float = 10,
) -> APIRouter:
    _reader_cache: dict[str, EnetFallMatDataSource] = {}
    last_alert_time = 0.0

    def _reader(scene: str) -> EnetFallMatDataSource:
        if scene not in _reader_cache:
            ds_names = SCENES.get(scene, DEFAULT_ENETFALL_DATASETS)
            _reader_cache[scene] = EnetFallMatDataSource.create_reader(dataset_names=ds_names)
        return _reader_cache[scene]

    @router.post("/trigger")
    async def trigger(req: DemoTriggerRequest) -> dict[str, Any]:
        nonlocal last_alert_time

        # ── Validate ────────────────────────────────────────────
        if req.scene not in SCENES:
            raise HTTPException(status_code=400, detail=f"Unknown scene '{req.scene}'. Choices: {list(SCENES)}")

        reader = _reader(req.scene)
        total = reader.total_samples
        idx = req.sample_index % total

        # ── Get window + label ──────────────────────────────────
        window = reader.get_window_at(idx)                        # [1, 3, 625, 30]
        if window is None:
            raise HTTPException(status_code=400, detail=f"Index {idx} out of range ({total} samples)")
        true_label = reader.get_label_at(idx)
        room_name = reader.get_room_at(idx) or req.room

        # ── Subcarrier preview ──────────────────────────────────
        # Average across antennas + time → [30] subcarrier slice
        w_arr = window.squeeze(0).detach().cpu().numpy()          # [3, 625, 30]
        preview = w_arr.mean(axis=(0, 1)).tolist()                # [30]
        if len(preview) < 1:
            preview = [0.0]

        # ── Build frame ─────────────────────────────────────────
        frame = CsiFrame(
            frame_id=runtime.total_frames + 1,
            device_id=req.device_id, timestamp=time.time(), room=room_name,
            subcarriers=preview, simulated_label=true_label, source="demo_trigger",
            window_shape=[3, 625, 30], label=true_label,
        )

        # ── Run detector ────────────────────────────────────────
        result = detector.predict_window(frame, window)

        # ── Analytics for the trigger window ────────────────────
        analytics_dict: dict[str, Any] | None = None
        try:
            raw = compute_analytics(window.squeeze(0))
            analytics_dict = raw
        except Exception:
            logger.warning("Analytics failed for demo trigger", exc_info=True)

        # ── Build evidence chain ────────────────────────────────
        before = req.evidence_before
        after = req.evidence_after
        chain: list[dict[str, Any]] = []
        for offset in range(-before, after + 1):
            ci = (idx + offset) % total
            w = reader.get_window_at(ci)
            entry: dict[str, Any] = {
                "window_index": ci,
                "room": reader.get_room_at(ci) or room_name,
                "label": reader.get_label_at(ci),
                "confidence": None,
                "analytics": None,
                "predicted_label": reader.get_label_at(ci),
            }
            if w is not None:
                try:
                    entry["analytics"] = compute_analytics(w.squeeze(0))
                except Exception:
                    pass
            chain.append(entry)

        # ── Persist alert ───────────────────────────────────────
        alert_saved = False
        event_id: str | None = None
        if result.alert:
            now = time.time()
            if now - last_alert_time >= alert_cooldown_seconds:
                try:
                    from app.db.database import SessionLocal
                    db = SessionLocal()
                    try:
                        alert_in = AlertEventCreate(
                            timestamp=result.timestamp, room=result.room,
                            device_id=frame.device_id,
                            predicted_label=result.predicted_label,
                            confidence=result.confidence, risk_level=result.risk_level,
                            activity_score=result.activity_score, reason=result.reason,
                            analytics_snapshot=analytics_dict, frame_id=frame.frame_id,
                        )
                        saved = alert_service.create_alert(db, alert_in)
                        event_id = saved.event_id
                        alert_saved = True
                        last_alert_time = now
                        logger.info("Demo alert saved: %s", event_id)
                    finally:
                        db.close()
                except Exception:
                    logger.warning("Alert persistence failed", exc_info=True)

        # ── Store evidence chain for replay API ─────────────────
        if event_id:
            runtime.store_evidence_chain(frame.frame_id, chain)

        # ── Register frame in runtime ───────────────────────────
        runtime.add(frame, result, AnalyticsSnapshot(**analytics_dict) if analytics_dict else None)

        # ── Broadcast ───────────────────────────────────────────
        envelope = {
            "frame": frame.model_dump(),
            "result": result.model_dump(),
            "avatar": _avatar(frame, result),
            "summary": runtime.get_summary(),
            "alert_saved": alert_saved,
            "analytics": analytics_dict,
            "event_id": event_id,
        }
        await ws_manager.broadcast(envelope)

        logger.info(
            "Demo trigger: scene=%s idx=%d/%d true=%s pred=%s conf=%.2f alert=%s",
            req.scene, idx, total, true_label,
            result.predicted_label, result.confidence, result.alert,
        )

        return {
            "frame": frame.model_dump(),
            "result": result.model_dump(),
            "alert_saved": alert_saved,
            "event_id": event_id,
            "sample_index": idx,
            "total_samples": total,
            "true_label": true_label,
            "evidence_chain": chain,
        }

    @router.get("/scenes")
    def list_scenes() -> dict[str, Any]:
        out: dict[str, dict[str, Any]] = {}
        for key in SCENES:
            try:
                r = _reader(key)
                fall_count = int(getattr(r, "labels", np.array([])).sum())
                out[key] = {"total_samples": r.total_samples, "fall_count": fall_count}
            except Exception:
                out[key] = {"total_samples": "unavailable"}
        return {"scenes": out}

    return router
