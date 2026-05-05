"""Utilidades de configuración del proyecto."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración general de la aplicación.

    La configuración se carga desde variables de entorno o, en su defecto,
    desde los valores por defecto definidos en esta clase.
    """

    environment: str = Field(default="development", alias="ENVIRONMENT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    data_raw_path: Path = Field(
        default=Path("data/raw/BankChurners.csv"),
        alias="DATA_RAW_PATH",
    )
    data_interim_path: Path = Field(
        default=Path("data/interim/churn_base.parquet"),
        alias="DATA_INTERIM_PATH",
    )
    data_validated_path: Path = Field(
        default=Path("data/processed/churn_validated.parquet"),
        alias="DATA_VALIDATED_PATH",
    )

    model_dir: Path = Field(default=Path("models"), alias="MODEL_DIR")
    sklearn_model_path: Path = Field(
        default=Path("models/gradient_boosting_tuned_pipeline.joblib"),
        alias="SKLEARN_MODEL_PATH",
    )

    pytorch_model_path: Path = Field(
        default=Path("models/churn_mlp_state_dict.pt"),
        alias="PYTORCH_MODEL_PATH",
    )
    pytorch_preprocessor_path: Path = Field(
        default=Path("models/pytorch_preprocessor.joblib"),
        alias="PYTORCH_PREPROCESSOR_PATH",
    )

    target_column: str = Field(default="Attrition_Flag", alias="TARGET_COLUMN")
    random_state: int = Field(default=42, alias="RANDOM_STATE")
    test_size: float = Field(default=0.2, alias="TEST_SIZE")

    mlflow_tracking_uri: str = Field(default="mlruns", alias="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field(
        default="credit-card-churn-scoring",
        alias="MLFLOW_EXPERIMENT_NAME",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def get_settings() -> Settings:
    """Devuelve la configuración general del proyecto."""
    return Settings()
