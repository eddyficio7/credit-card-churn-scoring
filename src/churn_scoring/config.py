"""Project configuration utilities."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    environment: str = Field(default="development", alias="ENVIRONMENT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    data_raw_path: Path = Field(
        default=Path("data/raw/credit_card_customers.csv"),
        alias="DATA_RAW_PATH",
    )
    data_processed_path: Path = Field(
        default=Path("data/processed/credit_card_customers_processed.parquet"),
        alias="DATA_PROCESSED_PATH",
    )

    model_dir: Path = Field(default=Path("models"), alias="MODEL_DIR")
    mlflow_tracking_uri: str = Field(default="mlruns", alias="MLFLOW_TRACKING_URI")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def get_settings() -> Settings:
    """Return project settings."""
    return Settings()
