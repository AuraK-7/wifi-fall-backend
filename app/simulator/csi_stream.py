from datetime import datetime, timezone

import numpy as np

from app.core.config import settings
from app.schemas.csi import CsiFrame


class CsiStreamSimulator:
    def __init__(self, device_id: str = "sim-device-001") -> None:
        self.device_id = device_id
        self.subcarrier_count = settings.CSI_SUBCARRIER_COUNT

    def next_frame(self) -> CsiFrame:
        amplitudes = np.random.normal(
            loc=1.0,
            scale=0.08,
            size=self.subcarrier_count,
        )
        phase = np.random.uniform(
            low=-np.pi,
            high=np.pi,
            size=self.subcarrier_count,
        )

        return CsiFrame(
            timestamp=datetime.now(timezone.utc),
            device_id=self.device_id,
            amplitudes=amplitudes.round(4).tolist(),
            phase=phase.round(4).tolist(),
        )
