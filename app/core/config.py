from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ENETFALL_DIR = BASE_DIR / "data" / "ENetFall_dataset_trained_networks"


class Settings(BaseSettings):
    APP_NAME: str = "WiFi Fall Guard"
    APP_ENV: str = "dev"
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DATABASE_URL: str = "sqlite:///./wifi_fall_guard.db"
    DETECTOR_MODE: str = "enetfall"
    ENETFALL_DATA_DIR: str = str(DEFAULT_ENETFALL_DIR)
    ENETFALL_MODEL_PATH: str = str(
        DEFAULT_ENETFALL_DIR / "B0(modified)_trained_with_all_data.pth"
    )

    CSI_FRAME_INTERVAL_MS: int = 100
    CSI_SUBCARRIER_COUNT: int = 64
    CSI_WINDOW_SIZE: int = 30
    DEFAULT_ROOM: str = "bedroom"

    FALL_CONFIDENCE_THRESHOLD: float = 0.75
    HIGH_ENERGY_THRESHOLD: float = 30.0
    LOW_ACTIVITY_THRESHOLD: float = 2.0

    ENABLE_FAKE_LABEL: bool = True

    API_VERSION: str = "0.1.0"
    API_PREFIX: str = "/api"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


settings = Settings()
