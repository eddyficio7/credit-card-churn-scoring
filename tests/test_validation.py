"""Pruebas para la validación de datos con Pandera."""

import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from churn_scoring.data.validation import validate_churn_data


def make_valid_churn_sample() -> pd.DataFrame:
    """Construye un dataset válido mínimo para probar el schema."""
    return pd.DataFrame(
        {
            "Attrition_Flag": ["Existing Customer", "Attrited Customer"],
            "Customer_Age": [45, 50],
            "Gender": ["M", "F"],
            "Dependent_count": [2, 1],
            "Education_Level": ["Graduate", "High School"],
            "Marital_Status": ["Married", "Single"],
            "Income_Category": ["$60K - $80K", "Less than $40K"],
            "Card_Category": ["Blue", "Silver"],
            "Months_on_book": [36, 48],
            "Total_Relationship_Count": [5, 4],
            "Months_Inactive_12_mon": [1, 2],
            "Contacts_Count_12_mon": [2, 3],
            "Credit_Limit": [3000.0, 5000.0],
            "Total_Revolving_Bal": [1000.0, 1500.0],
            "Avg_Open_To_Buy": [2000.0, 3500.0],
            "Total_Amt_Chng_Q4_Q1": [1.2, 0.8],
            "Total_Trans_Amt": [2400.0, 3600.0],
            "Total_Trans_Ct": [24, 36],
            "Total_Ct_Chng_Q4_Q1": [0.9, 1.1],
            "Avg_Utilization_Ratio": [0.3, 0.4],
            "Avg_Trans_Amt": [100.0, 100.0],
            "Products_Per_Month": [5 / 36, 4 / 48],
            "Contacts_Per_Inactive_Month": [2 / 2, 3 / 3],
        }
    )


def test_validate_churn_data_accepts_valid_dataframe() -> None:
    """Valida que un dataset correcto pase el schema."""
    valid_df = make_valid_churn_sample()

    result_df = validate_churn_data(valid_df)

    assert result_df.shape == valid_df.shape


def test_validate_churn_data_rejects_invalid_target() -> None:
    """Valida que el schema rechace etiquetas inválidas del target."""
    invalid_df = make_valid_churn_sample()
    invalid_df.loc[0, "Attrition_Flag"] = "Unknown Customer"

    with pytest.raises((SchemaError, SchemaErrors)):
        validate_churn_data(invalid_df)


def test_validate_churn_data_rejects_invalid_credit_limit_consistency() -> None:
    """Valida que el schema rechace inconsistencias en límite de crédito."""
    invalid_df = make_valid_churn_sample()
    invalid_df.loc[0, "Credit_Limit"] = 9999.0

    with pytest.raises((SchemaError, SchemaErrors)):
        validate_churn_data(invalid_df)


def test_validate_churn_data_rejects_duplicate_rows() -> None:
    """Valida que el schema rechace filas duplicadas."""
    valid_df = make_valid_churn_sample()
    duplicated_df = pd.concat([valid_df, valid_df.iloc[[0]]], ignore_index=True)

    with pytest.raises((SchemaError, SchemaErrors)):
        validate_churn_data(duplicated_df)
