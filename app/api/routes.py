from fastapi import APIRouter

from app.schemas.csi import CsiFrame, FallDetectionResult
from app.services.detector import FallDetector
from app.simulator.csi_stream import CsiStreamSimulator

router = APIRouter()
simulator = CsiStreamSimulator()
detector = FallDetector()


@router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "wifi-fall"}


@router.get("/csi/sample", response_model=CsiFrame)
def get_csi_sample() -> CsiFrame:
    return simulator.next_frame()


@router.get("/fall/sample", response_model=FallDetectionResult)
def get_fall_detection_sample() -> FallDetectionResult:
    frame = simulator.next_frame()
    return detector.detect(frame)
