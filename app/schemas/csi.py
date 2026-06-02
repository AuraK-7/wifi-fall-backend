from typing import Any, Literal

from pydantic import BaseModel, Field


ActivityLabel = Literal["empty", "walking", "sitting", "lying", "fall", "non_fall", "unknown"]
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
    source: str = "csv"
    window_shape: list[int] | None = None
    label: ActivityLabel | None = None


class DetectionResult(BaseModel):
    timestamp: float
    room: str
    predicted_label: ActivityLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    risk_level: RiskLevel
    alert: bool
    reason: str = ""
    activity_score: float = Field(0.0, ge=0.0, le=1.0)
    features: dict[str, Any] = Field(default_factory=dict)


class AnalyticsSnapshot(BaseModel):
    """Pre-computed signal features for multi-view visualisation."""
    micro_doppler_spectrum: list[float] = Field(..., min_length=128, max_length=128)
    subcarrier_amplitudes: list[float] = Field(..., min_length=1, max_length=90)
    antenna_correlation: float = Field(..., ge=-1.0, le=1.0)
    energy: float = Field(..., ge=0.0)
    dominant_freq: float
    frequency_spread: float = Field(..., ge=0.0)
    signal_variance: float = Field(..., ge=0.0)


class CsiStreamMessage(BaseModel):
    frame: CsiFrame
    result: DetectionResult
    analytics: AnalyticsSnapshot | None = None


class SimulatorCommand(BaseModel):
    label: ActivityLabel
    room: str | None = None


class CsvDataSourceCommand(BaseModel):
    csv_path: str
    room: str = "real_room"
    device_id: str = "csv-node-001"
    label: ActivityLabel = "unknown"


class EnetFallDataSourceCommand(BaseModel):
    data_dir: str | None = None
    dataset_names: list[str] | None = None
    device_id: str = "enetfall-node-001"
    room: str = "home"


class DetectorModeCommand(BaseModel):
    mode: Literal["simple", "enetfall", "cnn2d"]


class RecentResultItem(BaseModel):
    frame: CsiFrame
    result: DetectionResult
