from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class CsiFrame(BaseModel):
    timestamp: datetime
    device_id: str
    amplitudes: list[float] = Field(..., min_length=1)
    phase: list[float] = Field(..., min_length=1)


class FallDetectionResult(BaseModel):
    timestamp: datetime
    device_id: str
    score: float = Field(..., ge=0.0, le=1.0)
    status: Literal["normal", "fall_suspected"]
    message: str
