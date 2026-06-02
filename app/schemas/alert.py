from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AlertEventCreate(BaseModel):
    timestamp: float
    room: str
    device_id: str
    predicted_label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    risk_level: str
    activity_score: float = Field(..., ge=0.0, le=1.0)
    reason: str = ""
    analytics_snapshot: dict[str, Any] | None = None
    frame_id: int | None = None
    evidence_chain: list[dict[str, Any]] | None = None


class AlertEventRead(BaseModel):
    event_id: str
    timestamp: float
    room: str
    device_id: str
    predicted_label: str
    confidence: float
    risk_level: str
    activity_score: float
    reason: str | None
    handled: bool
    handler_note: str | None
    analytics_snapshot: dict[str, Any] | None = None
    frame_id: int | None = None
    evidence_chain: list[dict[str, Any]] | None = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AlertEventUpdate(BaseModel):
    handled: bool | None = None
    handler_note: str | None = None
