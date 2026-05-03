"""Pruebas para las funciones de limpieza y feature engineering."""

import pandas as pd

from churn_scoring.features.engineering import (
    NAIVE_BAYES_COLUMNS,
    add_churn_features,
    build_churn_base_dataset,
    drop_unnecessary_columns,
    encode_target,
)


def make_raw_sample() -> pd.DataFrame:
    """Construye un dataset crudo mínimo para pruebas."""
    return pd.DataFrame(
        {
            "CLIENTNUM": [1, 2],
            "Attrition_Flag": ["Existing Customer", "Attrited Customer"],
            "Total_Trans_Amt": [1000.0, 2400.0],
            "Total_Trans_Ct": [10, 24],
            "Total_Relationship_Count": [5, 6],
            "Months_on_book": [50, 60],
            "Contacts_Count_12_mon": [2, 4],
            "Months_Inactive_12_mon": [1, 3],
            NAIVE_BAYES_COLUMNS[0]: [0.1, 0.2],
            NAIVE_BAYES_COLUMNS[1]: [0.9, 0.8],
        }
    )


def test_drop_unnecessary_columns_removes_id_and_naive_bayes_columns() -> None:
    """Valida que se eliminen columnas no aptas para modelado."""
    raw_df = make_raw_sample()

    result_df = drop_unnecessary_columns(raw_df)

    assert "CLIENTNUM" not in result_df.columns
    assert NAIVE_BAYES_COLUMNS[0] not in result_df.columns
    assert NAIVE_BAYES_COLUMNS[1] not in result_df.columns


def test_add_churn_features_creates_expected_columns() -> None:
    """Valida que se creen correctamente las variables derivadas."""
    raw_df = make_raw_sample()
    clean_df = drop_unnecessary_columns(raw_df)

    result_df = add_churn_features(clean_df)

    assert "Avg_Trans_Amt" in result_df.columns
    assert "Products_Per_Month" in result_df.columns
    assert "Contacts_Per_Inactive_Month" in result_df.columns

    assert result_df.loc[0, "Avg_Trans_Amt"] == 100.0
    assert result_df.loc[0, "Products_Per_Month"] == 0.1
    assert result_df.loc[0, "Contacts_Per_Inactive_Month"] == 1.0


def test_build_churn_base_dataset_applies_cleaning_and_features() -> None:
    """Valida el flujo completo de construcción del dataset base."""
    raw_df = make_raw_sample()

    result_df = build_churn_base_dataset(raw_df)

    assert "CLIENTNUM" not in result_df.columns
    assert "Avg_Trans_Amt" in result_df.columns
    assert "Products_Per_Month" in result_df.columns
    assert "Contacts_Per_Inactive_Month" in result_df.columns


def test_encode_target_maps_labels_to_binary_values() -> None:
    """Valida la codificación binaria de la variable objetivo."""
    target = pd.Series(["Existing Customer", "Attrited Customer"])

    encoded = encode_target(target)

    assert encoded.tolist() == [0, 1]
