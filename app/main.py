import asyncio
import logging
import time
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
import numpy as np
import scipy.signal
import torch
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.db import models
from app.db.database import (
    Base,
    SessionLocal,
    engine,
    ensure_sqlite_schema_compatibility,
    get_db,
)
from app.schemas.alert import AlertEventCreate, AlertEventRead, AlertEventUpdate
from app.schemas.csi import (
    AnalyticsSnapshot,
    CsiFrame,
    CsvDataSourceCommand,
    DetectionResult,
    DetectorModeCommand,
    EnetFallDataSourceCommand,
)
from app.services.alert import AlertService
from app.services.enetfall_detector import ENetFallDetector
from app.services.data_source_manager import DataSourceManager
from app.data_sources.enetfall_mat_source import EnetFallMatDataSource
from app.services.detector import SimpleFallDetector
from app.services.runtime_state import RuntimeState
from app.services.signal_processor import compute_analytics

setup_logging()
logger = logging.getLogger(__name__)

Base.metadata.create_all(bind=engine)
ensure_sqlite_schema_compatibility()
logger.info("Application database tables initialized")

ALERT_COOLDOWN_SECONDS = 10

data_source_manager = DataSourceManager()
simple_detector = SimpleFallDetector()
enetfall_detector = ENetFallDetector()
detector_mode = settings.DETECTOR_MODE if settings.DETECTOR_MODE in {"simple", "enetfall"} else "enetfall"
runtime_state = RuntimeState()
alert_service = AlertService()
last_alert_time = 0.0

_ = models


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    logger.info("Application startup complete")
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.API_VERSION,
        description="Backend service for Wi-Fi CSI fall detection simulation.",
        lifespan=lifespan,
    )
    logger.info("FastAPI app created")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def root() -> dict[str, str]:
        return {
            "app": settings.APP_NAME,
            "env": settings.APP_ENV,
            "status": "running",
        }

    @app.get("/api/status")
    def get_status() -> dict[str, Any]:
        return {
            "app": settings.APP_NAME,
            "env": settings.APP_ENV,
            "source": data_source_manager.get_status(),
            "runtime": runtime_state.get_summary(),
        }

    @app.post("/api/data-source/csv")
    def switch_to_csv_source(command: CsvDataSourceCommand) -> dict[str, Any]:
        try:
            data_source_manager.switch_to_csv(
                csv_path=command.csv_path,
                room=command.room,
                device_id=command.device_id,
                label=command.label,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        logger.info("Data source switched to csv: %s", command.csv_path)
        return {
            "message": "Data source switched to csv",
            "source": data_source_manager.get_status(),
        }

    @app.post("/api/data-source/enetfall")
    def switch_to_enetfall_source(command: EnetFallDataSourceCommand) -> dict[str, Any]:
        try:
            data_source_manager.switch_to_enetfall(
                data_dir=command.data_dir,
                dataset_names=command.dataset_names,
                device_id=command.device_id,
                room=command.room,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        logger.info("Data source switched to ENetFall MAT replay")
        return {
            "message": "Data source switched to ENetFall MAT replay",
            "source": data_source_manager.get_status(),
        }

    @app.get("/api/data-source/status")
    def get_data_source_status() -> dict[str, Any]:
        return data_source_manager.get_status()

    @app.get("/api/model/status")
    def get_model_status() -> dict[str, Any]:
        status = enetfall_detector.get_status()
        status["active_detector_mode"] = detector_mode
        return status

    @app.post("/api/detector/mode")
    def update_detector_mode(command: DetectorModeCommand) -> dict[str, Any]:
        global detector_mode
        detector_mode = command.mode
        logger.info("Detector mode changed to %s", detector_mode)
        return {
            "message": "Detector mode updated",
            "mode": detector_mode,
            "model": enetfall_detector.get_status(),
        }

    @app.get("/api/results/latest")
    def get_latest_result() -> dict[str, Any]:
        latest = runtime_state.get_latest()
        if latest["frame"] is None or latest["result"] is None:
            raise HTTPException(status_code=404, detail="No CSI result available")
        latest["avatar"] = _avatar_payload(latest["frame"], latest["result"])
        return latest

    @app.get("/api/results/recent")
    def get_recent_results(limit: int = Query(default=50, ge=1, le=300)) -> list[dict[str, Any]]:
        items = runtime_state.get_recent(limit=limit)
        return [
            {
                "frame": item.frame,
                "result": item.result,
                "avatar": _avatar_payload(item.frame, item.result),
            }
            for item in items
        ]

    @app.get("/api/sequences")
    def get_sequence(
        activity_type: str = Query("fall"),
        sample_index: int = Query(0),
        downsample_step: int = Query(1)
    ) -> Any:
        source = data_source_manager.get_current_source()
        if not hasattr(source, "processed_tensor") or not hasattr(source, "labels"):
            raise HTTPException(status_code=400, detail="Current data source does not support sequence extraction. Suggest switching to ENetFall.")
        
        target_label = 1 if activity_type.lower() == "fall" else 0
        indices = np.where(source.labels == target_label)[0]
        
        if len(indices) == 0:
            raise HTTPException(status_code=404, detail=f"No samples found for activity {activity_type}")
            
        actual_index = indices[sample_index % len(indices)]
        
        window = source.processed_tensor[actual_index].numpy() 
        window = window - np.mean(window, axis=1, keepdims=True)

        f, t, Zxx = scipy.signal.stft(window, fs=100, axis=1, nperseg=64, noverlap=56)
        spectrogram = np.mean(np.abs(Zxx), axis=(0, 2))[1:, :]
        
        spectrogram = np.clip(spectrogram * 10, 0, 1.0)

        frames_out = []
        for time_idx in range(spectrogram.shape[1]):
            if time_idx % downsample_step != 0 and time_idx != spectrogram.shape[1] - 1:
                continue
                
            amp = spectrogram[:, time_idx].tolist()
            amp_rounded = [round(float(v), 4) for v in amp]
            
            energy = sum((v**2) for v in amp_rounded)
            mean_val = sum(amp_rounded) / len(amp_rounded)
            variance = sum((v - mean_val)**2 for v in amp_rounded) / len(amp_rounded)
            
            frames_out.append({
                "t": time_idx,
                "amplitude": amp_rounded,
                "energy": round(energy, 4),
                "variance": round(variance, 4)
            })

        flat_amp = spectrogram.flatten()
        return {
            "metadata": {
                "activity_type": activity_type,
                "true_label": "fall" if target_label == 1 else "walk",
                "true_label_id": target_label,
                "avatar_state": _label_to_avatar_state("fall" if target_label == 1 else "non_fall"),
                "sample_index": sample_index,
                "total_samples_of_type": int(len(indices)),
                "total_frames_raw": int(spectrogram.shape[1]),
                "total_frames_downsampled": len(frames_out),
                "downsample_step": downsample_step,
                "subcarrier_count": int(spectrogram.shape[0]),
                "amplitude_min": round(float(np.min(flat_amp)), 4),
                "amplitude_max": round(float(np.max(flat_amp)), 4),
                "amplitude_mean": round(float(np.mean(flat_amp)), 4),
                "amplitude_std": round(float(np.std(flat_amp)), 4),
            },
            "frames": frames_out
        }

    @app.get("/api/events/{event_id}/replay")
    def get_event_replay(
        event_id: str,
        before: int = Query(default=120, ge=20, le=300),
        after: int = Query(default=40, ge=10, le=150),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        """Return analytics windows around an alert event for 3D replay.

        Finds the nearest non_fall→fall transition BEFORE the alert frame,
        then returns a long stretch of walking (before) leading into the fall.
        """
        alert = alert_service.get_alert(db, event_id)
        if alert is None:
            raise HTTPException(status_code=404, detail="Alert event not found")

        chain = runtime_state.get_evidence_chain(alert.frame_id or 0)
        if chain:
            # Map evidence entries to frontend format: predicted_label → label
            windows = []
            for entry in chain:
                windows.append({
                    "window_index": entry.get("window_index", 0),
                    "room": alert.room,
                    "analytics": entry.get("analytics"),
                    "label": entry.get("predicted_label", "unknown"),
                    "confidence": entry.get("confidence"),
                })
            return {
                "event_id": event_id,
                "start_window_index": windows[0]["window_index"] if windows else 0,
                "end_window_index": windows[-1]["window_index"] if windows else 0,
                "centre_window_index": alert.frame_id or 0,
                "window_count": len(windows),
                "windows": windows,
            }

        raise HTTPException(status_code=404, detail="No evidence chain — replay unavailable for this alert")

    @app.post("/api/detector/reset")
    def reset_detector() -> dict[str, str]:
        simple_detector.reset()
        enetfall_detector.reset()
        global runtime_state
        runtime_state = RuntimeState()
        return {"message": "Detector and runtime state reset"}


    @app.get("/api/window/{frame_id}")
    def get_window_analytics(frame_id: int) -> dict[str, Any]:
        """Return the analytics snapshot for a specific frame_id."""
        snap = runtime_state.get_analytics_by_frame_id(frame_id)
        if snap is None:
            raise HTTPException(
                status_code=404,
                detail=f"No analytics found for frame {frame_id}",
            )
        return {"frame_id": frame_id, "analytics": snap}

    @app.get("/api/event/{event_id}/windows")
    def get_event_windows(
        event_id: str,
        before: int = Query(default=100, ge=0, le=300),
        after: int = Query(default=100, ge=0, le=300),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        """Return analytics snapshots around an alert event's frame."""
        alert = alert_service.get_alert(db, event_id)
        if alert is None:
            raise HTTPException(status_code=404, detail="Alert event not found")
        centre_fid = _find_closest_frame_id(alert.timestamp)
        windows = runtime_state.get_analytics_window(centre_fid, before, after)

        return {
            "event_id": event_id,
            "alert_timestamp": alert.timestamp,
            "centre_frame_id": centre_fid,
            "window_count": len(windows),
            "windows": windows,
        }
    @app.get("/api/alerts", response_model=list[AlertEventRead])
    def list_alerts(
        skip: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=300),
        handled: bool | None = None,
        db: Session = Depends(get_db),
    ) -> list[Any]:
        return alert_service.list_alerts(db=db, skip=skip, limit=limit, handled=handled)

    @app.get("/api/alerts/summary/count")
    def get_alert_count_summary(db: Session = Depends(get_db)) -> dict[str, int]:
        total = alert_service.count_alerts(db)
        handled = alert_service.count_alerts(db, handled=True)
        unhandled = alert_service.count_alerts(db, handled=False)
        return {
            "total": total,
            "handled": handled,
            "unhandled": unhandled,
        }

    @app.get("/api/alerts/{event_id}", response_model=AlertEventRead)
    def get_alert(event_id: str, db: Session = Depends(get_db)) -> Any:
        alert = alert_service.get_alert(db, event_id)
        if alert is None:
            raise HTTPException(status_code=404, detail="Alert event not found")
        return alert

    @app.patch("/api/alerts/{event_id}", response_model=AlertEventRead)
    def update_alert(
        event_id: str,
        update_in: AlertEventUpdate,
        db: Session = Depends(get_db),
    ) -> Any:
        alert = alert_service.update_alert(db, event_id, update_in)
        if alert is None:
            raise HTTPException(status_code=404, detail="Alert event not found")

        if update_in.handled is not None:
            logger.info("Alert %s handled status changed to %s", event_id, update_in.handled)
        return alert

    @app.websocket("/ws/csi")
    async def stream_csi(websocket: WebSocket) -> None:
        await websocket.accept()
        logger.info("WebSocket CSI client connected")

        try:
            while True:
                try:
                    frame, result, window = _next_detection()

                    analytics: AnalyticsSnapshot | None = None
                    analytics_dict: dict[str, Any] | None = None
                    if window is not None:
                        try:
                            raw = compute_analytics(window.squeeze(0))
                            analytics = AnalyticsSnapshot(**raw)
                            analytics_dict = analytics.model_dump()
                        except Exception:
                            logger.warning(
                                "Analytics computation failed for frame %s",
                                frame.frame_id,
                                exc_info=True,
                            )

                    runtime_state.add(frame, result, analytics)

                    if result.alert:
                        runtime_state.start_evidence_chain(frame.frame_id)

                    alert_saved = False
                    try:
                        alert_saved = (
                            save_alert_if_needed(result, frame, analytics_dict) is not None
                        )
                    except Exception:
                        logger.warning(
                            "Alert persistence failed for frame %s",
                            frame.frame_id,
                            exc_info=True,
                        )

                    await websocket.send_json(
                        {
                            "frame": frame.model_dump(),
                            "result": result.model_dump(),
                            "avatar": _avatar_payload(frame, result),
                            "summary": runtime_state.get_summary(),
                            "alert_saved": alert_saved,
                            "analytics": analytics_dict,
                        }
                    )
                except WebSocketDisconnect:
                    logger.info("WebSocket CSI client disconnected")
                    break
                except RuntimeError as e:
                    if "Cannot call" in str(e) or "close" in str(e):
                        logger.info("WebSocket connection closed internally")
                        break
                    logger.error("Error generating next detection: %s", e)
                except Exception as e:
                    logger.error("Error generating next detection: %s", e)
                await asyncio.sleep(settings.CSI_FRAME_INTERVAL_MS / 1000)
        except WebSocketDisconnect:
            logger.info("WebSocket CSI client disconnected")
        except Exception as e:
            logger.info("WebSocket terminated: %s", e)
        return

    return app


def _next_detection() -> tuple[CsiFrame, DetectionResult, torch.Tensor | None]:
    """Return (frame, result, window_tensor_or_None).

    The window tensor [1, 3, 625, 30] is needed for analytics computation.
    In simple-detector mode this is ``None`` because no window is available.
    """
    source = data_source_manager.get_current_source()
    if detector_mode == "enetfall" and hasattr(source, "next_window"):
        frame, window, _ = source.next_window()
        result = enetfall_detector.predict_window(frame, window)
        return frame, result, window

    frame = source.next_frame()
    result = simple_detector.predict(frame)
    return frame, result, None


def _avatar_payload(frame: CsiFrame, result: DetectionResult) -> dict[str, Any]:
    """Map dataset/model labels to the two states supported by the 3D avatar."""
    dataset_label = frame.label or frame.simulated_label
    predicted_label = result.predicted_label
    dataset_state = _label_to_avatar_state(dataset_label)
    predicted_state = _label_to_avatar_state(predicted_label)

    return {
        "display_state": dataset_state,
        "dataset_state": dataset_state,
        "predicted_state": predicted_state,
        "source": "dataset_label",
        "dataset_label": dataset_label,
        "predicted_label": predicted_label,
        "confidence": result.confidence,
        "risk_level": result.risk_level,
        "alert": result.alert,
    }


def _label_to_avatar_state(label: str | None) -> str:
    if label == "fall":
        return "fallen"
    return "standing"



def _find_closest_frame_id(timestamp: float) -> int:
    """Find the analytics buffer frame_id closest to the given unix timestamp."""
    return runtime_state.find_closest_frame_id(timestamp)


def save_alert_if_needed(
    result: DetectionResult,
    frame: CsiFrame,
    analytics: dict[str, Any] | None = None,
    
) -> Any | None:
    if not result.alert:
        return None

    now = time.time()
    global last_alert_time
    if now - last_alert_time < ALERT_COOLDOWN_SECONDS:
        return None

    db = SessionLocal()
    try:
        alert_in = AlertEventCreate(
            timestamp=result.timestamp,
            room=result.room,
            device_id=frame.device_id,
            predicted_label=result.predicted_label,
            confidence=result.confidence,
            risk_level=result.risk_level,
            activity_score=result.activity_score,
            reason=result.reason,
            analytics_snapshot=analytics,
            frame_id=frame.frame_id,
        )
        alert = alert_service.create_alert(db, alert_in)
        last_alert_time = now
        logger.info("Alert saved successfully: %s", alert.event_id)
        return alert
    finally:
        db.close()


app = create_app()
