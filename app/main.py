import asyncio
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.schemas.csi import ActivityLabel
from app.services.detector import SimpleFallDetector
from app.services.runtime_state import RuntimeState
from app.simulator.csi_stream import CsiStreamSimulator

VALID_LABELS: set[str] = {"empty", "walking", "sitting", "lying", "fall", "unknown"}

simulator = CsiStreamSimulator()
detector = SimpleFallDetector()
runtime_state = RuntimeState()


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
    def get_status() -> dict[str, Any]:
        return {
            "app": settings.APP_NAME,
            "env": settings.APP_ENV,
            "simulator": {
                "current_label": simulator.current_label,
                "room": simulator.room,
                "device_id": simulator.device_id,
                "subcarrier_count": simulator.subcarrier_count,
                "frame_interval_ms": settings.CSI_FRAME_INTERVAL_MS,
                "sequence_enabled": bool(simulator.sequence),
                "sequence_loop": simulator.sequence_loop,
            },
            "runtime": runtime_state.get_summary(),
        }

    @app.post("/api/simulator/label/{label}")
    def update_simulator_label(label: str) -> dict[str, str]:
        active_label = _parse_label(label)
        simulator.set_label(active_label)
        return {
            "message": "Simulator label updated",
            "current_label": active_label,
        }

    @app.post("/api/simulator/room/{room}")
    def update_simulator_room(room: str) -> dict[str, str]:
        if not room.strip():
            raise HTTPException(status_code=400, detail="Room cannot be empty")

        simulator.set_room(room)
        return {
            "message": "Simulator room updated",
            "room": simulator.room,
        }

    @app.post("/api/simulator/device/{device_id}")
    def update_simulator_device(device_id: str) -> dict[str, str]:
        if not device_id.strip():
            raise HTTPException(status_code=400, detail="Device id cannot be empty")

        simulator.set_device(device_id)
        return {
            "message": "Simulator device updated",
            "device_id": simulator.device_id,
        }

    @app.post("/api/simulator/sequence")
    def load_simulator_sequence(sequence: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            simulator.load_sequence(sequence)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return {
            "message": "Simulator sequence loaded",
            "sequence_length": len(simulator.sequence),
            "sequence_loop": simulator.sequence_loop,
        }

    @app.delete("/api/simulator/sequence")
    def clear_simulator_sequence() -> dict[str, str]:
        simulator.clear_sequence()
        return {"message": "Simulator sequence cleared"}

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

    @app.websocket("/ws/csi")
    async def stream_csi(websocket: WebSocket) -> None:
        await websocket.accept()

        try:
            while True:
                frame = simulator.next_frame()
                result = detector.predict(frame)
                runtime_state.add(frame, result)
                await websocket.send_json(
                    {
                        "frame": frame.model_dump(),
                        "result": result.model_dump(),
                        "summary": runtime_state.get_summary(),
                    }
                )
                await asyncio.sleep(settings.CSI_FRAME_INTERVAL_MS / 1000)
        except WebSocketDisconnect:
            return

    return app


def _parse_label(label: str) -> ActivityLabel:
    if label not in VALID_LABELS:
        raise HTTPException(status_code=400, detail="Invalid simulator label")
    return cast(ActivityLabel, label)


app = create_app()
