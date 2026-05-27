from typing import Literal

from pydantic import BaseModel, Field


ActivityLabel = Literal["empty", "walking", "sitting", "lying", "fall", "unknown"]
RiskLevel = Literal["low", "medium", "high"]


class DeviceInfo(BaseModel):
    device_id: str
    room: str
    device_type: str = "simulated_csi_node"
    online: bool = True


class CsiFrame(BaseModel):
    frame_id: int = Field(..., ge=0)
    device_id: str = "sim-node-001"
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
    activity_score: float = Field(0.0, ge=0.0, le=1.0)
    features: dict[str, float] = Field(default_factory=dict)


class CsiStreamMessage(BaseModel):
    frame: CsiFrame
    result: DetectionResult


class SimulatorCommand(BaseModel):
    label: ActivityLabel
    room: str | None = None


class RecentResultItem(BaseModel):
    frame: CsiFrame
    result: DetectionResult
