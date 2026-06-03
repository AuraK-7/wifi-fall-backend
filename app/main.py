import asyncio
import hashlib
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
from pydantic import BaseModel, Field, ValidationError
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
from app.schemas.demo import (
    DemoCsiEnvelope,
    DemoCsiPacket,
    DemoPacketAck,
    MobileFallEventCreate,
    MobileFallEventResponse,
    MobileModelConfig,
)
from app.services.alert import AlertService
from app.services.connection_manager import ConnectionManager
from app.services.enetfall_detector import ENetFallDetector
from app.services.cnn2d_detector import CNN2DFallDetector
from app.services.data_source_manager import DataSourceManager
from app.data_sources.enetfall_mat_source import EnetFallMatDataSource
from app.services.detector import SimpleFallDetector
from app.services.runtime_state import RuntimeState
from app.services.signal_processor import compute_analytics
from app.api.demo import create_demo_router

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


ws_manager = ConnectionManager()


class MobileCsiConnectionManager:
    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self._connections.discard(websocket)

    async def broadcast_packet(self, packet: DemoCsiPacket) -> None:
        message = DemoCsiEnvelope(payload=packet).model_dump(mode="json")
        stale_connections: list[WebSocket] = []
        for websocket in list(self._connections):
            try:
                await websocket.send_json(message)
            except Exception:
                stale_connections.append(websocket)

        for websocket in stale_connections:
            self.disconnect(websocket)


mobile_csi_connections = MobileCsiConnectionManager()


class ModelActivateCommand(BaseModel):
    model_id: str | None = None
    path: str | None = None
    detector_type: str | None = None

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

    @app.websocket("/ws/demo/source")
    async def receive_demo_source(websocket: WebSocket) -> None:
        await websocket.accept()
        logger.info("Demo source WebSocket connected")
        try:
            while True:
                try:
                    data = await websocket.receive_json()
                    packet = _parse_demo_source_message(data)
                    ack = _demo_packet_ack(packet)
                    await mobile_csi_connections.broadcast_packet(packet)
                    await websocket.send_json(ack.model_dump(mode="json"))
                except ValidationError as exc:
                    await websocket.send_json(
                        {
                            "accepted": False,
                            "queued_at": time.time(),
                            "message": str(exc),
                        }
                    )
        except WebSocketDisconnect:
            logger.info("Demo source WebSocket disconnected")

    @app.post("/api/demo/packets", response_model=DemoPacketAck)
    async def submit_demo_packet(packet: DemoCsiPacket) -> DemoPacketAck:
        ack = _demo_packet_ack(packet)
        await mobile_csi_connections.broadcast_packet(packet)
        return ack

    @app.websocket("/ws/mobile/csi")
    async def stream_mobile_csi(websocket: WebSocket) -> None:
        await mobile_csi_connections.connect(websocket)
        logger.info("Mobile CSI WebSocket connected")
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            logger.info("Mobile CSI WebSocket disconnected")
        finally:
            mobile_csi_connections.disconnect(websocket)

    @app.get("/api/mobile/model-config", response_model=MobileModelConfig)
    def get_mobile_model_config() -> MobileModelConfig:
        return MobileModelConfig()

    @app.post("/api/mobile/fall-events", response_model=MobileFallEventResponse)
    def save_mobile_fall_event(
        event_in: MobileFallEventCreate,
        db: Session = Depends(get_db),
    ) -> MobileFallEventResponse:
        existing = alert_service.get_alert(db, event_in.event_id)
        if existing is None:
            alert = models.AlertEvent(
                event_id=event_in.event_id,
                timestamp=event_in.timestamp,
                room=event_in.room,
                device_id=event_in.device_id,
                predicted_label=event_in.result.predicted_label,
                confidence=event_in.result.confidence,
                risk_level=event_in.result.risk_level,
                activity_score=event_in.result.activity_score,
                reason=event_in.result.reason,
                analytics_snapshot=_mobile_event_snapshot(event_in),
                frame_id=event_in.packet.frame_id,
                evidence_chain=_mobile_event_evidence_chain(event_in),
                handled=False,
            )
            db.add(alert)
            db.commit()
            db.refresh(alert)
            logger.info("Mobile fall event saved: %s", alert.event_id)

        return MobileFallEventResponse(
            event_id=event_in.event_id,
            saved=True,
            replay_url=f"#/replay?eventId={event_in.event_id}",
        )

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

    @app.get("/api/models")
    def list_models() -> dict[str, Any]:
        return _model_list_payload()

    @app.get("/api/model/list")
    def list_models_compat() -> dict[str, Any]:
        return _model_list_payload()

    @app.post("/api/model/activate")
    def activate_model(command: ModelActivateCommand) -> dict[str, Any]:
        selected = _find_discovered_model(command)
        if selected is None:
            raise HTTPException(status_code=404, detail="Model was not found in configured model paths")

        selected_type = command.detector_type or selected["detector_type"]
        if selected_type not in {"cnn2d", "enetfall"}:
            raise HTTPException(
                status_code=400,
                detail="detector_type must be cnn2d or enetfall for this model",
            )

        _activate_model_path(selected["path"], selected_type)
        return {
            "message": "Model activated",
            "model": selected,
            "active_detector_mode": detector_mode,
            "status": (
                cnn2d_detector.get_status()
                if detector_mode == "cnn2d"
                else enetfall_detector.get_status()
            ),
        }

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

        Finds the nearest non_fall to fall transition BEFORE the alert frame,
        then returns a long stretch of walking (before) leading into the fall.
        """
        alert = alert_service.get_alert(db, event_id)
        if alert is None:
            raise HTTPException(status_code=404, detail="Alert event not found")

        chain = runtime_state.get_evidence_chain(alert.frame_id or 0)
        if chain:
            return _replay_payload_from_chain(
                event_id=event_id,
                room=alert.room,
                chain=chain,
                centre_window_index=alert.frame_id or 0,
            )

        if alert.evidence_chain:
            return _replay_payload_from_chain(
                event_id=event_id,
                room=alert.room,
                chain=alert.evidence_chain,
                centre_window_index=_stored_replay_centre(alert.evidence_chain),
            )

        raise HTTPException(status_code=404, detail="No evidence chain - replay unavailable for this alert")

    @app.post("/api/detector/reset")
    def reset_detector() -> dict[str, str]:
        simple_detector.reset()
        enetfall_detector.reset()
        cnn2d_detector.reset()
        global runtime_state
        runtime_state = RuntimeState()
        return {"message": "Detector and runtime state reset"}

    # ------------------------------------------------------------
    # Training Job Management
    # ------------------------------------------------------------

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
                        detail=f"Training job {j['job_id'][:8]} is already running.",
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
                raise HTTPException(status_code=500, detail=f"Failed to start training process: {exc}") from exc

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
        """Apply a completed training job: copy model + normalizer to active paths, reload detector."""
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
            logger.info("Applied model %s -> %s", src_model, dst_model)

            # Copy normalizer stats if available
            if src_norm.exists():
                dst_norm_dir = Path(settings.CNN2D_NORMALIZER_DIR)
                dst_norm_dir.mkdir(parents=True, exist_ok=True)
                _shutil.copy2(src_norm, dst_norm_dir / "csi_zscore_stats.json")
                logger.info("Applied normalizer %s -> %s", src_norm, dst_norm_dir)

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
        """Return per-room model evaluation metrics from the last training run.

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

        listen_only = websocket.query_params.get("mode") == "demo"
        client_queue = ws_manager.register()
        logger.info(
            "WebSocket CSI client connected (mode=%s)",
            "demo-listen" if listen_only else "replay",
        )

        replay_task: asyncio.Task[None] | None = None

        async def _replay_loop() -> None:
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

                    await client_queue.put(
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
                    raise
                except RuntimeError as e:
                    if "Cannot call" in str(e) or "close" in str(e):
                        raise WebSocketDisconnect from e
                    logger.error("Error generating next detection: %s", e)
                except Exception as e:
                    logger.error("Error generating next detection: %s", e)
                await asyncio.sleep(settings.CSI_FRAME_INTERVAL_MS / 1000)

        try:
            if listen_only:
                # Listen-only mode: wait for demo-trigger broadcasts
                while True:
                    message = await client_queue.get()
                    try:
                        await websocket.send_json(message)
                    except Exception:
                        break
            else:
                # Replay mode: start the replay loop and forward messages
                replay_task = asyncio.create_task(_replay_loop())
                while True:
                    message = await client_queue.get()
                    try:
                        await websocket.send_json(message)
                    except Exception:
                        break
        except WebSocketDisconnect:
            logger.info("WebSocket CSI client disconnected")
        except Exception as e:
            logger.info("WebSocket terminated: %s", e)
        finally:
            if replay_task is not None:
                replay_task.cancel()
            ws_manager.unregister(client_queue)
        return

    # ── Demo single-trigger router (ENetFall) ────────────────
    demo_router = create_demo_router(
        detector=enetfall_detector,
        runtime=runtime_state,
        ws_manager=ws_manager,
        alert_service=alert_service,
        alert_cooldown_seconds=ALERT_COOLDOWN_SECONDS,
    )
    app.include_router(demo_router)

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


def _model_list_payload() -> dict[str, Any]:
    status = (
        cnn2d_detector.get_status()
        if detector_mode == "cnn2d"
        else enetfall_detector.get_status()
    )
    active_model_path = _normalize_model_path(status.get("model_path"))
    search_sources = _model_search_sources()
    extensions = _model_file_extensions()
    models = _discover_model_files(search_sources, extensions, active_model_path)

    return {
        "active_detector_mode": detector_mode,
        "active_model_path": status.get("model_path"),
        "extensions": sorted(extensions),
        "search_paths": [
            {
                "env_key": env_key,
                "path": str(path),
                "exists": path.exists(),
                "is_dir": path.is_dir(),
            }
            for env_key, path in search_sources
        ],
        "models": models,
    }


def _model_file_extensions() -> set[str]:
    raw = settings.MODEL_FILE_EXTENSIONS or ".pt,.pth"
    extensions = {
        item.strip().lower()
        for item in raw.replace(";", ",").split(",")
        if item.strip()
    }
    return {
        item if item.startswith(".") else f".{item}"
        for item in extensions
    }


def _find_discovered_model(command: ModelActivateCommand) -> dict[str, Any] | None:
    payload = _model_list_payload()
    target_path = _normalize_model_path(command.path)
    for model in payload["models"]:
        if command.model_id and model["model_id"] == command.model_id:
            return model
        if target_path and _normalize_model_path(model["path"]) == target_path:
            return model
    return None


def _activate_model_path(model_path: str, detector_type: str) -> None:
    global detector_mode, cnn2d_detector, enetfall_detector
    if detector_type == "cnn2d":
        detector = CNN2DFallDetector(model_path=model_path)
        if not detector.model_loaded:
            raise HTTPException(status_code=400, detail=detector.load_error or "CNN2D model failed to load")
        cnn2d_detector = detector
        detector_mode = "cnn2d"
        return
    if detector_type == "enetfall":
        detector = ENetFallDetector(model_path=model_path)
        if not detector.model_loaded:
            raise HTTPException(status_code=400, detail=detector.load_error or "ENetFall model failed to load")
        enetfall_detector = detector
        detector_mode = "enetfall"
        return
    raise ValueError(f"Unsupported detector_type: {detector_type}")


def _model_search_sources() -> list[tuple[str, Path]]:
    sources: list[tuple[str, Path]] = []
    for item in _split_model_search_paths(settings.MODEL_SEARCH_PATHS):
        sources.append(("MODEL_SEARCH_PATHS", Path(item)))

    configured_paths = {
        "ENETFALL_MODEL_PATH": settings.ENETFALL_MODEL_PATH,
        "CNN2D_MODEL_PATH": settings.CNN2D_MODEL_PATH,
        "ENETFALL_DATA_DIR": settings.ENETFALL_DATA_DIR,
    }
    normalizer_parent = Path(settings.CNN2D_NORMALIZER_DIR).parent
    configured_paths["CNN2D_NORMALIZER_PARENT"] = str(normalizer_parent)

    for env_key, raw_path in configured_paths.items():
        if raw_path:
            sources.append((env_key, Path(raw_path)))

    return _dedupe_model_sources(sources)


def _split_model_search_paths(raw: str) -> list[str]:
    if not raw:
        return []
    normalized = raw.replace("\n", ";")
    parts: list[str] = []
    for chunk in normalized.split(";"):
        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)
    return parts


def _dedupe_model_sources(sources: list[tuple[str, Path]]) -> list[tuple[str, Path]]:
    seen: set[str] = set()
    deduped: list[tuple[str, Path]] = []
    for env_key, path in sources:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((env_key, path))
    return deduped


def _discover_model_files(
    sources: list[tuple[str, Path]],
    extensions: set[str],
    active_model_path: str | None = None,
) -> list[dict[str, Any]]:
    seen_files: set[str] = set()
    models: list[dict[str, Any]] = []
    for env_key, source_path in sources:
        for model_path in _iter_model_files(source_path, extensions):
            normalized = _normalize_model_path(str(model_path))
            if normalized is None or normalized in seen_files:
                continue
            seen_files.add(normalized)
            stat = model_path.stat()
            models.append(
                {
                    "model_id": hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12],
                    "file_name": model_path.name,
                    "path": str(model_path),
                    "source_env_key": env_key,
                    "detector_type": _guess_detector_type(model_path),
                    "extension": model_path.suffix.lower(),
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(
                        stat.st_mtime,
                        tz=timezone.utc,
                    ).isoformat(),
                    "active": normalized == active_model_path,
                }
            )

    return sorted(models, key=lambda item: (item["detector_type"], item["file_name"]))


def _iter_model_files(source_path: Path, extensions: set[str]) -> list[Path]:
    try:
        if source_path.is_file():
            directory = source_path.parent
            direct_file = (
                [source_path]
                if source_path.suffix.lower() in extensions
                else []
            )
        else:
            directory = source_path
            direct_file = []

        if not directory.exists() or not directory.is_dir():
            return direct_file

        return direct_file + [
            child
            for child in directory.iterdir()
            if child.is_file() and child.suffix.lower() in extensions
        ]
    except OSError:
        return []


def _normalize_model_path(raw_path: Any) -> str | None:
    if not raw_path:
        return None
    path = Path(str(raw_path))
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _guess_detector_type(model_path: Path) -> str:
    name = model_path.name.lower()
    parent = str(model_path.parent).lower()
    if "cnn" in name or "checkpoint" in parent:
        return "cnn2d"
    if "b0" in name or "efficientnet" in name or "enetfall" in parent:
        return "enetfall"
    return "unknown"


def _parse_demo_source_message(data: Any) -> DemoCsiPacket:
    if isinstance(data, dict) and data.get("type") == "demo_csi_packet" and "payload" in data:
        return DemoCsiEnvelope.model_validate(data).payload
    return DemoCsiPacket.model_validate(data)


def _demo_packet_ack(packet: DemoCsiPacket) -> DemoPacketAck:
    return DemoPacketAck(
        accepted=True,
        packet_id=packet.packet_id,
        sequence_id=packet.sequence_id,
        queued_at=time.time(),
        message="queued",
    )


def _mobile_event_snapshot(event_in: MobileFallEventCreate) -> dict[str, Any]:
    return {
        "source": "mobile_fall_event",
        "packet_id": event_in.packet_id,
        "sequence_id": event_in.sequence_id,
        "model": event_in.model.model_dump(mode="json"),
        "packet": event_in.packet.model_dump(mode="json"),
        "result": event_in.result.model_dump(mode="json"),
        "analytics": event_in.analytics,
        "avatar": event_in.result.avatar.model_dump(mode="json"),
    }


def _mobile_event_evidence_chain(event_in: MobileFallEventCreate) -> list[dict[str, Any]]:
    frames = event_in.packet.window
    if not frames:
        frames = [
            {
                "frame_index": 0,
                "timestamp": event_in.packet.timestamp,
                "subcarriers": event_in.packet.subcarriers,
                "energy": event_in.result.energy,
                "variance": event_in.result.variance,
            }
        ]

    avatar = event_in.result.avatar.model_dump(mode="json")
    chain: list[dict[str, Any]] = []
    for fallback_index, frame in enumerate(frames):
        frame_data = (
            frame.model_dump(mode="json")
            if hasattr(frame, "model_dump")
            else dict(frame)
        )
        window_index = int(frame_data.get("frame_index", fallback_index))
        chain.append(
            {
                "window_index": window_index,
                "room": event_in.room,
                "timestamp": frame_data.get("timestamp", event_in.timestamp),
                "analytics": _mobile_window_analytics(event_in, frame_data),
                "label": event_in.result.predicted_label,
                "predicted_label": event_in.result.predicted_label,
                "confidence": event_in.result.confidence,
                "avatar": avatar,
            }
        )
    return chain


def _mobile_window_analytics(
    event_in: MobileFallEventCreate,
    frame_data: dict[str, Any],
) -> dict[str, Any]:
    analytics = dict(event_in.analytics or {})
    subcarriers = frame_data.get("subcarriers") or event_in.packet.subcarriers
    energy = frame_data.get("energy")
    variance = frame_data.get("variance")

    analytics["subcarrier_amplitudes"] = subcarriers
    analytics["energy"] = energy if energy is not None else event_in.result.energy or analytics.get("energy", 0.0)
    analytics["signal_variance"] = (
        variance
        if variance is not None
        else event_in.result.variance or analytics.get("signal_variance", 0.0)
    )
    analytics.setdefault("micro_doppler_spectrum", [])
    analytics.setdefault("antenna_correlation", 0.0)
    analytics.setdefault("dominant_freq", 0.0)
    analytics.setdefault("frequency_spread", 0.0)
    return analytics


def _replay_payload_from_chain(
    event_id: str,
    room: str,
    chain: list[dict[str, Any]],
    centre_window_index: int,
) -> dict[str, Any]:
    windows: list[dict[str, Any]] = []
    for entry in chain:
        label = entry.get("label") or entry.get("predicted_label", "unknown")
        windows.append(
            {
                "window_index": entry.get("window_index", 0),
                "room": entry.get("room", room),
                "analytics": entry.get("analytics"),
                "label": label,
                "confidence": entry.get("confidence"),
                "avatar": entry.get("avatar") or _avatar_from_replay_entry(entry),
            }
        )

    return {
        "event_id": event_id,
        "start_window_index": windows[0]["window_index"] if windows else 0,
        "end_window_index": windows[-1]["window_index"] if windows else 0,
        "centre_window_index": centre_window_index,
        "window_count": len(windows),
        "windows": windows,
    }


def _stored_replay_centre(chain: list[dict[str, Any]]) -> int:
    if not chain:
        return 0
    centre_offset = min(40, len(chain) - 1)
    return int(chain[centre_offset].get("window_index", centre_offset))


def _avatar_from_replay_entry(entry: dict[str, Any]) -> dict[str, Any]:
    predicted_label = entry.get("predicted_label") or entry.get("label", "unknown")
    predicted_state = _label_to_avatar_state(predicted_label)
    return {
        "display_state": predicted_state,
        "predicted_state": predicted_state,
        "source": "model_prediction",
        "predicted_label": predicted_label,
        "confidence": entry.get("confidence"),
        "risk_level": "high" if predicted_label == "fall" else "low",
        "alert": predicted_label == "fall",
    }


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
            source="replay",
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
