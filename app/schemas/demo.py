from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


AvatarState = Literal["standing", "fallen", "unknown"]


class DemoCsiWindowFrame(BaseModel):
    frame_index: int = Field(..., ge=0)
    timestamp: float | None = None
    subcarriers: list[float] = Field(..., min_length=1)
    energy: float | None = None
    variance: float | None = None

    model_config = ConfigDict(extra="allow")


class DemoCsiPacket(BaseModel):
    packet_id: str
    sequence_id: str | None = None
    frame_id: int = Field(..., ge=0)
    timestamp: float
    room: str = "demo_room"
    device_id: str = "console-csi-001"
    source: str = "console"
    mode: Literal["single", "stream"] = "single"
    subcarrier_count: int = Field(..., ge=1)
    window_size: int = Field(..., ge=1)
    subcarriers: list[float] = Field(default_factory=list)
    window: list[DemoCsiWindowFrame] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    @classmethod
    def reject_ground_truth_labels(cls, data: Any) -> Any:
        if isinstance(data, dict):
            forbidden = {"label", "simulated_label", "dataset_label", "true_label"}
            present = sorted(forbidden.intersection(data))
            if present:
                raise ValueError(
                    "Demo CSI packets must not include ground-truth labels: "
                    + ", ".join(present)
                )
        return data


class DemoCsiEnvelope(BaseModel):
    type: Literal["demo_csi_packet"] = "demo_csi_packet"
    payload: DemoCsiPacket


class DemoPacketAck(BaseModel):
    accepted: bool
    packet_id: str
    sequence_id: str | None = None
    queued_at: float
    message: str = "queued"


class MobileModelConfig(BaseModel):
    runtime: Literal["onnx-web", "tflite", "mock"] = "mock"
    weight_url: str = "/models/mobile-fall.onnx"
    input_shape: list[int] = Field(default_factory=lambda: [1, 64, 30])
    class_names: list[str] = Field(default_factory=lambda: ["non_fall", "fall"])
    threshold: float = Field(0.75, ge=0.0, le=1.0)


class MobileAvatarPayload(BaseModel):
    display_state: AvatarState
    dataset_state: AvatarState = "unknown"
    predicted_state: AvatarState
    source: str = "mobile_model"
    dataset_label: str | None = None
    predicted_label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    risk_level: str
    alert: bool

    model_config = ConfigDict(extra="allow")


class MobileInferenceResult(BaseModel):
    predicted_label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    risk_level: str
    alert: bool
    activity_score: float = Field(0.0, ge=0.0, le=1.0)
    energy: float | None = None
    variance: float | None = None
    reason: str = ""
    avatar: MobileAvatarPayload

    model_config = ConfigDict(extra="allow")


class MobileFallEventCreate(BaseModel):
    event_id: str
    packet_id: str
    sequence_id: str | None = None
    timestamp: float
    room: str
    device_id: str = "mobile-detector-001"
    model: MobileModelConfig
    packet: DemoCsiPacket
    result: MobileInferenceResult
    analytics: dict[str, Any] | None = None

    model_config = ConfigDict(extra="allow")


class MobileFallEventResponse(BaseModel):
    event_id: str
    saved: bool
    replay_url: str
    message: str = "saved"
