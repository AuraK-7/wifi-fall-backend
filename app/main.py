import asyncio
import logging
import time
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.db import models
from app.db.database import Base, SessionLocal, engine, get_db
from app.schemas.alert import AlertEventCreate, AlertEventRead, AlertEventUpdate
from app.schemas.csi import CsiFrame, CsvDataSourceCommand, DetectionResult
from app.services.alert import AlertService
from app.services.data_source_manager import DataSourceManager
from app.services.detector import SimpleFallDetector
from app.services.runtime_state import RuntimeState

setup_logging()
logger = logging.getLogger(__name__)

Base.metadata.create_all(bind=engine)
logger.info("Application database tables initialized")

ALERT_COOLDOWN_SECONDS = 10

data_source_manager = DataSourceManager()
detector = SimpleFallDetector()
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

    @app.get("/api/data-source/status")
    def get_data_source_status() -> dict[str, Any]:
        return data_source_manager.get_status()

    @app.get("/api/results/latest")
    def get_latest_result() -> dict[str, Any]:
        latest = runtime_state.get_latest()
        if latest["frame"] is None or latest["result"] is None:
            raise HTTPException(status_code=404, detail="No CSI result available")
        return latest

    @app.get("/api/results/recent")
    def get_recent_results(limit: int = Query(default=50, ge=1, le=300)) -> list[Any]:
        return runtime_state.get_recent(limit=limit)

    @app.post("/api/detector/reset")
    def reset_detector() -> dict[str, str]:
        detector.reset()
        global runtime_state
        runtime_state = RuntimeState()
        return {"message": "Detector and runtime state reset"}

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
                frame = data_source_manager.get_current_source().next_frame()
                result = detector.predict(frame)
                runtime_state.add(frame, result)
                alert_saved = save_alert_if_needed(result, frame) is not None
                await websocket.send_json(
                    {
                        "frame": frame.model_dump(),
                        "result": result.model_dump(),
                        "summary": runtime_state.get_summary(),
                        "alert_saved": alert_saved,
                    }
                )
                await asyncio.sleep(settings.CSI_FRAME_INTERVAL_MS / 1000)
        except WebSocketDisconnect:
            logger.info("WebSocket CSI client disconnected")
            return

    return app


def save_alert_if_needed(result: DetectionResult, frame: CsiFrame) -> Any | None:
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
        )
        alert = alert_service.create_alert(db, alert_in)
        last_alert_time = now
        logger.info("Alert saved successfully: %s", alert.event_id)
        return alert
    finally:
        db.close()


app = create_app()
