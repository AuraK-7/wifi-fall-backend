from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "WiFi Fall Guard"
    APP_ENV: str = "dev"
    HOST: str = "127.0.0.1"
    PORT: int = 8000

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
