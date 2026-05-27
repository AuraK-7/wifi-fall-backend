from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "wifi-fall"
    API_VERSION: str = "0.1.0"
    API_PREFIX: str = "/api"
    ENVIRONMENT: str = "development"
    CORS_ORIGINS: list[str] = ["http://localhost:5173"]

    CSI_SAMPLE_RATE_HZ: int = 20
    CSI_SUBCARRIER_COUNT: int = 30
    CSI_STREAM_INTERVAL_SECONDS: float = 0.1
    FALL_THRESHOLD: float = 0.75

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


settings = Settings()
