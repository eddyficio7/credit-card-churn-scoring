"""Utilidades de inferencia para la aplicación de churn."""

from dataclasses import dataclass

import pandas as pd

from churn_scoring.features.engineering import add_churn_features
from churn_scoring.models.base import BaseChurnModel, ChurnPrediction
from churn_scoring.models.sklearn_model import assign_recommendation, assign_risk_level

MODEL_FEATURE_COLUMNS = [
    "Customer_Age",
    "Gender",
    "Dependent_count",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
    "Avg_Trans_Amt",
    "Products_Per_Month",
    "Contacts_Per_Inactive_Month",
]


@dataclass(frozen=True)
class CustomerInput:
    """Representa los datos de entrada de un cliente para inferencia.

    Esta estructura usa nombres en español/snake_case para facilitar su uso
    dentro de la aplicación, pero después los convierte a las columnas esperadas
    por el pipeline de modelado.
    """

    customer_age: int
    gender: str
    dependent_count: int
    education_level: str
    marital_status: str
    income_category: str
    card_category: str
    months_on_book: int
    total_relationship_count: int
    months_inactive_12_mon: int
    contacts_count_12_mon: int
    credit_limit: float
    total_revolving_bal: float
    avg_open_to_buy: float
    total_amt_chng_q4_q1: float
    total_trans_amt: float
    total_trans_ct: int
    total_ct_chng_q4_q1: float
    avg_utilization_ratio: float

    def to_raw_dataframe(self) -> pd.DataFrame:
        """Convierte el input del cliente a DataFrame con columnas originales.

        Returns
        -------
        pd.DataFrame
            DataFrame de una fila con los nombres de columnas del dataset.
        """
        return pd.DataFrame(
            [
                {
                    "Customer_Age": self.customer_age,
                    "Gender": self.gender,
                    "Dependent_count": self.dependent_count,
                    "Education_Level": self.education_level,
                    "Marital_Status": self.marital_status,
                    "Income_Category": self.income_category,
                    "Card_Category": self.card_category,
                    "Months_on_book": self.months_on_book,
                    "Total_Relationship_Count": self.total_relationship_count,
                    "Months_Inactive_12_mon": self.months_inactive_12_mon,
                    "Contacts_Count_12_mon": self.contacts_count_12_mon,
                    "Credit_Limit": self.credit_limit,
                    "Total_Revolving_Bal": self.total_revolving_bal,
                    "Avg_Open_To_Buy": self.avg_open_to_buy,
                    "Total_Amt_Chng_Q4_Q1": self.total_amt_chng_q4_q1,
                    "Total_Trans_Amt": self.total_trans_amt,
                    "Total_Trans_Ct": self.total_trans_ct,
                    "Total_Ct_Chng_Q4_Q1": self.total_ct_chng_q4_q1,
                    "Avg_Utilization_Ratio": self.avg_utilization_ratio,
                }
            ]
        )

    def to_model_dataframe(self) -> pd.DataFrame:
        """Construye el DataFrame final esperado por el modelo.

        Returns
        -------
        pd.DataFrame
            DataFrame con variables originales y derivadas, en el orden esperado
            por el pipeline de scikit-learn.
        """
        raw_df = self.to_raw_dataframe()
        feature_df = add_churn_features(raw_df)

        return feature_df[MODEL_FEATURE_COLUMNS]


def predict_customer_churn(
    model: BaseChurnModel,
    customer_input: CustomerInput,
) -> ChurnPrediction:
    """Genera una predicción enriquecida de churn para un cliente.

    Parameters
    ----------
    model : BaseChurnModel
        Modelo de churn que cumple la interfaz común del proyecto.
    customer_input : CustomerInput
        Datos del cliente.

    Returns
    -------
    ChurnPrediction
        Predicción con clase, probabilidad, nivel de riesgo y recomendación.
    """
    input_df = customer_input.to_model_dataframe()

    prediction = int(model.predict(input_df).iloc[0])
    churn_probability = float(model.predict_proba(input_df).iloc[0])

    risk_level = assign_risk_level(churn_probability)
    recommendation = assign_recommendation(risk_level)

    return ChurnPrediction(
        prediction=prediction,
        churn_probability=churn_probability,
        risk_level=risk_level,
        recommendation=recommendation,
    )
