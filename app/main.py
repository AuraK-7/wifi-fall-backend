import asyncio
from typing import cast

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.schemas.csi import ActivityLabel, CsiStreamMessage
from app.services.detector import SimpleFallDetector
from app.simulator.csi_stream import CsiStreamSimulator

simulator = CsiStreamSimulator()
detector = SimpleFallDetector()

VALID_LABELS: set[str] = {"empty", "walking", "sitting", "lying", "fall", "unknown"}


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.API_VERSION,
        description="Backend service for Wi-Fi CSI fall detection simulation.",
    )

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
    def get_status() -> dict[str, str | int]:
        return {
            "current_label": simulator.current_label,
            "room": simulator.room,
            "subcarrier_count": simulator.subcarrier_count,
            "frame_interval_ms": settings.CSI_FRAME_INTERVAL_MS,
        }

    @app.post("/api/simulator/label/{label}")
    def update_simulator_label(label: str) -> dict[str, str]:
        if label not in VALID_LABELS:
            raise HTTPException(status_code=400, detail="Invalid simulator label")

        active_label = cast(ActivityLabel, label)
        simulator.set_label(active_label)
        return {
            "message": "Simulator label updated",
            "current_label": label,
        }

    @app.websocket("/ws/csi")
    async def stream_csi(websocket: WebSocket) -> None:
        await websocket.accept()

        try:
            while True:
                frame = simulator.next_frame()
                result = detector.predict(frame)
                message = CsiStreamMessage(frame=frame, result=result)
                await websocket.send_json(message.model_dump())
                await asyncio.sleep(settings.CSI_FRAME_INTERVAL_MS / 1000)
        except WebSocketDisconnect:
            return

    return app


app = create_app()
