import asyncio
import logging
import os as _os
import signal as _signal
import subprocess as _subprocess
import threading
import time
import uuid
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
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
from app.services.cnn2d_detector import CNN2DFallDetector
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
cnn2d_detector = CNN2DFallDetector()
VALID_MODES = {"simple", "enetfall", "cnn2d"}
detector_mode = settings.DETECTOR_MODE if settings.DETECTOR_MODE in VALID_MODES else "cnn2d"
runtime_state = RuntimeState()
alert_service = AlertService()
last_alert_time = 0.0

# ── Training job management  ─────────────────────────────
_TRAIN_SCRIPT = (Path(__file__).resolve().parents[1] / "train.py").as_posix()
_JOBS_DIR = (Path(__file__).resolve().parents[1] / "data" / "jobs")
_JOBS_DIR.mkdir(parents=True, exist_ok=True)
_training_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()

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
        if detector_mode == "cnn2d":
            status = cnn2d_detector.get_status()
        else:
            status = enetfall_detector.get_status()
        status["active_detector_mode"] = detector_mode
        return status

    @app.post("/api/detector/mode")
    def update_detector_mode(command: DetectorModeCommand) -> dict[str, Any]:
        global detector_mode
        if command.mode not in VALID_MODES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode '{command.mode}'. Valid: {sorted(VALID_MODES)}",
            )
        detector_mode = command.mode
        logger.info("Detector mode changed to %s", detector_mode)
        active = (
            cnn2d_detector if detector_mode == "cnn2d" else enetfall_detector
        )
        return {
            "message": "Detector mode updated",
            "mode": detector_mode,
            "model": active.get_status(),
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
        cnn2d_detector.reset()
        global runtime_state
        runtime_state = RuntimeState()
        return {"message": "Detector and runtime state reset"}

    # ═══════════════════════════════════════════════════════════════
    # Training Job Management
    # ═══════════════════════════════════════════════════════════════

    class TrainStartRequest(BaseModel):
        epochs: int = Field(default=200, ge=50, le=300)
        batch_size: int = Field(default=32, ge=8, le=128)
        lr: float = Field(default=0.0005, ge=0.0001, le=0.01)
        p_mix: float = Field(default=0.5, ge=0.0, le=1.0)
        p_shadow: float = Field(default=0.5, ge=0.0, le=1.0)
        p_stretch: float = Field(default=0.5, ge=0.0, le=1.0)
        p_noise: float = Field(default=0.5, ge=0.0, le=1.0)
        weight_decay: float = Field(default=1e-4, ge=1e-5, le=1e-3)

    @app.post("/api/train/start")
    def train_start(req: TrainStartRequest) -> dict[str, Any]:
        """Start an async training job. Returns 409 if one is already running."""
        with _jobs_lock:
            for j in _training_jobs.values():
                if j["status"] in ("pending", "running"):
                    raise HTTPException(
                        status_code=409,
                        detail=f"训练任务 {j['job_id'][:8]} 已在运行中，请等待完成或先停止",
                    )

            job_id = uuid.uuid4().hex[:12]
            job_dir = _JOBS_DIR / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            output_path = job_dir / "training_results.json"
            log_path = job_dir / "training.log"

            cmd = [
                _os.sys.executable, _TRAIN_SCRIPT,
                "--epochs", str(req.epochs),
                "--batch-size", str(req.batch_size),
                "--lr", str(req.lr),
                "--p-mix", str(req.p_mix),
                "--p-shadow", str(req.p_shadow),
                "--p-stretch", str(req.p_stretch),
                "--p-noise", str(req.p_noise),
                "--weight-decay", str(req.weight_decay),
                "--output", str(output_path),
                "--log-file", str(log_path),
            ]

            logger.info("Starting training job %s: %s", job_id, cmd)

            try:
                proc = _subprocess.Popen(
                    cmd,
                    stdout=_subprocess.DEVNULL,
                    stderr=_subprocess.STDOUT,
                    cwd=str(Path(_TRAIN_SCRIPT).parent),
                )
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"无法启动训练进程: {exc}") from exc

            job = {
                "job_id": job_id,
                "status": "running",
                "params": req.model_dump(),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": None,
                "output_path": str(output_path),
                "log_path": str(log_path),
                "pid": proc.pid,
                "error": None,
                "best_val_f1": None,
            }
            _training_jobs[job_id] = job

            # Background thread to wait for completion
            def _monitor(jid: str, p: _subprocess.Popen) -> None:
                ret = p.wait()
                with _jobs_lock:
                    j = _training_jobs.get(jid)
                    if j is None:
                        return
                    j["finished_at"] = datetime.now(timezone.utc).isoformat()
                    if ret == 0:
                        j["status"] = "completed"
                        try:
                            import json as _json
                            res = _json.loads(Path(j["output_path"]).read_text())
                            j["best_val_f1"] = res.get("best_val_f1")
                        except Exception:
                            pass
                    elif j["status"] == "stopped":
                        pass  # already marked
                    else:
                        j["status"] = "failed"
                        j["error"] = f"Process exited with code {ret}"

            threading.Thread(target=_monitor, args=(job_id, proc), daemon=True).start()

            return {"job_id": job_id, "status": "running"}

    @app.get("/api/train/status/{job_id}")
    def train_status(job_id: str) -> dict[str, Any]:
        with _jobs_lock:
            job = _training_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
            # Check if still running
            if job["status"] == "running" and job.get("pid"):
                try:
                    _os.kill(job["pid"], 0)  # check if alive
                except OSError:
                    job["status"] = "completed" if not job.get("error") else "failed"
            return dict(job)

    @app.get("/api/train/list")
    def train_list() -> list[dict[str, Any]]:
        with _jobs_lock:
            jobs = sorted(
                _training_jobs.values(),
                key=lambda j: j.get("started_at", ""),
                reverse=True,
            )
            # Return lightweight summary plus full running job
            result = []
            for j in jobs[:20]:
                result.append({
                    "job_id": j["job_id"],
                    "status": j["status"],
                    "params": j["params"],
                    "started_at": j["started_at"],
                    "finished_at": j["finished_at"],
                    "best_val_f1": j["best_val_f1"],
                    "error": j["error"],
                })
            return result

    @app.get("/api/train/log/{job_id}")
    def train_log(job_id: str, lines: int = Query(default=200, ge=10, le=2000)) -> dict[str, Any]:
        with _jobs_lock:
            job = _training_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
            log_path = job.get("log_path", "")
        if not log_path or not _os.path.exists(log_path):
            return {"job_id": job_id, "log": "", "lines": 0}
        try:
            text = Path(log_path).read_text(encoding="utf-8", errors="replace")
            all_lines = text.splitlines()
            tail = all_lines[-lines:] if len(all_lines) > lines else all_lines
            return {
                "job_id": job_id,
                "log": "\n".join(tail),
                "lines": len(tail),
                "total_lines": len(all_lines),
            }
        except Exception as exc:
            return {"job_id": job_id, "log": f"[read error: {exc}]", "lines": 0}

    @app.post("/api/train/stop/{job_id}")
    def train_stop(job_id: str) -> dict[str, Any]:
        with _jobs_lock:
            job = _training_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
            if job["status"] not in ("pending", "running"):
                raise HTTPException(status_code=400, detail=f"Job {job_id} is already {job['status']}")
            pid = job.get("pid")
            if pid is not None:
                try:
                    _os.kill(pid, _signal.SIGTERM)
                except OSError:
                    pass
            job["status"] = "stopped"
            job["finished_at"] = datetime.now(timezone.utc).isoformat()
            return {"job_id": job_id, "status": "stopped"}

    @app.post("/api/train/apply/{job_id}")
    def train_apply(job_id: str) -> dict[str, Any]:
        """Apply a completed training job: copy model + normalizer → active paths, reload detector."""
        with _jobs_lock:
            job = _training_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
            if job["status"] != "completed":
                raise HTTPException(
                    status_code=400,
                    detail=f"Job {job_id} is {job['status']}, only completed jobs can be applied",
                )

            job_dir = Path(job["output_path"]).parent
            src_model = job_dir / f"{Path(job['output_path']).stem}_best.pth"
            src_norm = job_dir / "normalizer" / "csi_zscore_stats.json"

            if not src_model.exists():
                raise HTTPException(status_code=404, detail=f"Model file not found: {src_model}")

            # Copy model to active path
            import shutil as _shutil
            dst_model = Path(settings.CNN2D_MODEL_PATH)
            dst_model.parent.mkdir(parents=True, exist_ok=True)
            _shutil.copy2(src_model, dst_model)
            logger.info("Applied model %s → %s", src_model, dst_model)

            # Copy normalizer stats if available
            if src_norm.exists():
                dst_norm_dir = Path(settings.CNN2D_NORMALIZER_DIR)
                dst_norm_dir.mkdir(parents=True, exist_ok=True)
                _shutil.copy2(src_norm, dst_norm_dir / "csi_zscore_stats.json")
                logger.info("Applied normalizer %s → %s", src_norm, dst_norm_dir)

            # Reload the CNN2D detector
            global cnn2d_detector
            try:
                cnn2d_detector = CNN2DFallDetector()
                loaded = cnn2d_detector.model_loaded
            except Exception as exc:
                logger.error("Failed to reload detector after apply: %s", exc)
                loaded = False

            # Also copy results JSON so /api/model/metrics picks it up
            dst_results = dst_model.parent / "training_results.json"
            try:
                _shutil.copy2(job["output_path"], dst_results)
            except Exception:
                pass

            job["applied"] = True
            return {
                "job_id": job_id,
                "applied": True,
                "model_loaded": loaded,
                "model_path": str(dst_model),
                "best_val_f1": job.get("best_val_f1"),
            }

    @app.get("/api/model/metrics")
    def get_model_metrics() -> dict[str, Any]:
        """Return per‑room model evaluation metrics from the last training run.

        Sources the ``training_results.json`` written by ``train.py``.
        Falls back to a live evaluation on the current data source if no
        results file exists.
        """
        import json as _json
        results_path = settings.CNN2D_MODEL_PATH.replace(
            "lightweight_2dcnn_best.pth", "training_results.json"
        )
        if not __import__("os").path.exists(results_path):
            # Try relative to BASE_DIR
            from app.core.config import BASE_DIR
            alt = BASE_DIR / "data" / "checkpoints" / "training_results.json"
            if alt.exists():
                results_path = str(alt)
        try:
            with open(results_path) as f:
                return _json.load(f)
        except FileNotFoundError:
            return {
                "error": "No training results found. Run train.py first.",
                "path_checked": results_path,
            }


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
    In cnn2d mode the raw [1, 625, 90] tensor is returned.
    """
    source = data_source_manager.get_current_source()

    if detector_mode == "cnn2d" and hasattr(source, "next_window_2d"):
        frame, window_2d, _ = source.next_window_2d()
        result = cnn2d_detector.predict(window_2d, frame)
        # Normalize the window for analytics display
        if cnn2d_detector._normalizer is not None:
            window_2d = cnn2d_detector._normalizer.normalize(window_2d)
        return frame, result, window_2d

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
