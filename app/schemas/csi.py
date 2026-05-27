from typing import Literal

from pydantic import BaseModel, Field


ActivityLabel = Literal["empty", "walking", "sitting", "lying", "fall", "unknown"]
RiskLevel = Literal["low", "medium", "high"]


class CsiFrame(BaseModel):
    timestamp: float
    room: str
    subcarriers: list[float] = Field(..., min_length=1)
    simulated_label: ActivityLabel = "unknown"


class DetectionResult(BaseModel):
    timestamp: float
    room: str
    predicted_label: ActivityLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    risk_level: RiskLevel
    alert: bool
    reason: str = ""


class CsiStreamMessage(BaseModel):
    frame: CsiFrame
    result: DetectionResult
