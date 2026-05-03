"""Construye y valida el dataset base de churn."""

from churn_scoring.config import get_settings
from churn_scoring.data.loader import load_csv_dataset, save_parquet_dataset
from churn_scoring.data.validation import validate_churn_data
from churn_scoring.features.engineering import build_churn_base_dataset


def main() -> None:
    """Ejecuta el flujo de construcción del dataset base."""
    settings = get_settings()

    raw_df = load_csv_dataset(settings.data_raw_path)
    churn_base = build_churn_base_dataset(raw_df)
    churn_validated = validate_churn_data(churn_base)

    save_parquet_dataset(churn_base, settings.data_interim_path)
    save_parquet_dataset(churn_validated, settings.data_validated_path)

    print(f"Dataset base guardado en: {settings.data_interim_path}")
    print(f"Dataset validado guardado en: {settings.data_validated_path}")


if __name__ == "__main__":
    main()
