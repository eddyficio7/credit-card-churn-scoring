"""Pruebas para las funciones de carga y guardado de datasets."""

from pathlib import Path

import pandas as pd
import pytest

from churn_scoring.data.loader import (
    load_csv_dataset,
    load_parquet_dataset,
    save_parquet_dataset,
)


def test_load_csv_dataset_returns_dataframe(tmp_path: Path) -> None:
    """Valida que un archivo CSV se cargue como DataFrame."""
    csv_path = tmp_path / "sample.csv"
    expected_df = pd.DataFrame(
        {
            "Customer_Age": [45, 50],
            "Credit_Limit": [1000.0, 2000.0],
        }
    )
    expected_df.to_csv(csv_path, index=False)

    result_df = load_csv_dataset(csv_path)

    pd.testing.assert_frame_equal(result_df, expected_df)


def test_load_csv_dataset_raises_file_not_found(tmp_path: Path) -> None:
    """Valida que se lance error cuando el CSV no existe."""
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        load_csv_dataset(missing_path)


def test_save_and_load_parquet_dataset(tmp_path: Path) -> None:
    """Valida que un DataFrame se guarde y cargue correctamente en Parquet."""
    parquet_path = tmp_path / "sample.parquet"
    expected_df = pd.DataFrame(
        {
            "Customer_Age": [45, 50],
            "Credit_Limit": [1000.0, 2000.0],
        }
    )

    save_parquet_dataset(expected_df, parquet_path)
    result_df = load_parquet_dataset(parquet_path)

    pd.testing.assert_frame_equal(result_df, expected_df)


def test_load_parquet_dataset_raises_file_not_found(tmp_path: Path) -> None:
    """Valida que se lance error cuando el archivo Parquet no existe."""
    missing_path = tmp_path / "missing.parquet"

    with pytest.raises(FileNotFoundError):
        load_parquet_dataset(missing_path)
