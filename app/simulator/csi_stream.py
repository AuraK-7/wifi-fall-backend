import time

import numpy as np

from app.core.config import settings
from app.schemas.csi import CsiFrame


class CsiStreamSimulator:
    def __init__(self, room: str = settings.DEFAULT_ROOM) -> None:
        self.room = room
        self.subcarrier_count = settings.CSI_SUBCARRIER_COUNT

    def next_frame(self) -> CsiFrame:
        subcarriers = np.random.normal(
            loc=1.0,
            scale=0.08,
            size=self.subcarrier_count,
        )

        return CsiFrame(
            timestamp=time.time(),
            room=self.room,
            subcarriers=subcarriers.round(4).tolist(),
            simulated_label="unknown",
        )
