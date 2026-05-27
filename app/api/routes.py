from fastapi import APIRouter

from app.schemas.csi import CsiFrame, DetectionResult
from app.services.detector import SimpleFallDetector
from app.simulator.csi_stream import CsiStreamSimulator

router = APIRouter()
simulator = CsiStreamSimulator()
detector = SimpleFallDetector()


@router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "wifi-fall"}


@router.get("/csi/sample", response_model=CsiFrame)
def get_csi_sample() -> CsiFrame:
    return simulator.next_frame()


@router.get("/fall/sample", response_model=DetectionResult)
def get_fall_detection_sample() -> DetectionResult:
    frame = simulator.next_frame()
    return detector.predict(frame)
