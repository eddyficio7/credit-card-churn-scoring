"""Pruebas para utilidades de inferencia de la aplicación."""

import pandas as pd

from churn_scoring.app.inference import (
    MODEL_FEATURE_COLUMNS,
    CustomerInput,
    predict_customer_churn,
)
from churn_scoring.models.base import BaseChurnModel, ChurnPrediction


class FakeChurnModel(BaseChurnModel):
    """Modelo falso para probar la lógica de inferencia de la app."""

    def predict(self, input_data: pd.DataFrame) -> pd.Series:
        """Devuelve una predicción fija."""
        return pd.Series([1], index=input_data.index, name="prediction")

    def predict_proba(self, input_data: pd.DataFrame) -> pd.Series:
        """Devuelve una probabilidad fija de churn."""
        return pd.Series(
            [0.75],
            index=input_data.index,
            name="churn_probability",
        )


def make_customer_input() -> CustomerInput:
    """Construye un input válido de cliente para pruebas."""
    return CustomerInput(
        customer_age=45,
        gender="M",
        dependent_count=2,
        education_level="Graduate",
        marital_status="Married",
        income_category="$60K - $80K",
        card_category="Blue",
        months_on_book=36,
        total_relationship_count=5,
        months_inactive_12_mon=1,
        contacts_count_12_mon=2,
        credit_limit=3000.0,
        total_revolving_bal=1000.0,
        avg_open_to_buy=2000.0,
        total_amt_chng_q4_q1=1.2,
        total_trans_amt=2400.0,
        total_trans_ct=24,
        total_ct_chng_q4_q1=0.9,
        avg_utilization_ratio=0.3,
    )


def test_customer_input_to_raw_dataframe() -> None:
    """Valida que el input se convierta a DataFrame original."""
    customer_input = make_customer_input()

    result_df = customer_input.to_raw_dataframe()

    assert result_df.shape == (1, 19)
    assert result_df.loc[0, "Customer_Age"] == 45
    assert result_df.loc[0, "Gender"] == "M"


def test_customer_input_to_model_dataframe_adds_engineered_features() -> None:
    """Valida que el DataFrame final incluya variables derivadas."""
    customer_input = make_customer_input()

    result_df = customer_input.to_model_dataframe()

    assert result_df.columns.tolist() == MODEL_FEATURE_COLUMNS
    assert result_df.loc[0, "Avg_Trans_Amt"] == 100.0
    assert result_df.loc[0, "Products_Per_Month"] == 5 / 36
    assert result_df.loc[0, "Contacts_Per_Inactive_Month"] == 1.0


def test_predict_customer_churn_returns_churn_prediction() -> None:
    """Valida que la inferencia regrese una predicción enriquecida."""
    customer_input = make_customer_input()
    model = FakeChurnModel()

    result = predict_customer_churn(model, customer_input)

    assert isinstance(result, ChurnPrediction)
    assert result.prediction == 1
    assert result.churn_probability == 0.75
    assert result.risk_level == "Alto"
    assert result.recommendation == "Priorizar acción de retención personalizada."
