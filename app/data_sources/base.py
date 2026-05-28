from abc import ABC, abstractmethod

from app.schemas.csi import ActivityLabel, CsiFrame


class BaseCsiSource(ABC):
    @abstractmethod
    def next_frame(self) -> CsiFrame:
        raise NotImplementedError

    @abstractmethod
    def set_label(self, label: ActivityLabel) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_room(self, room: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_device(self, device_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_status(self) -> dict:
        raise NotImplementedError
