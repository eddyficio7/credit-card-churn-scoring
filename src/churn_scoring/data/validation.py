"""Validación de datos para el dataset de churn con Pandera."""

import numpy as np
import pandas as pd
import pandera.pandas as pa


def check_credit_limit_consistency(df: pd.DataFrame) -> bool:
    """Valida la consistencia entre límite de crédito y saldo disponible.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset de churn.

    Returns
    -------
    bool
        True si el límite de crédito es consistente con el saldo revolvente
        y el monto disponible para compra.
    """
    return np.isclose(
        df["Credit_Limit"],
        df["Total_Revolving_Bal"] + df["Avg_Open_To_Buy"],
        rtol=1e-3,
        atol=1.0,
    ).all()


def check_avg_trans_amt(df: pd.DataFrame) -> bool:
    """Valida que la variable derivada Avg_Trans_Amt esté bien calculada.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset de churn con variables derivadas.

    Returns
    -------
    bool
        True si Avg_Trans_Amt coincide con Total_Trans_Amt / Total_Trans_Ct.
    """
    return np.isclose(
        df["Avg_Trans_Amt"],
        df["Total_Trans_Amt"] / df["Total_Trans_Ct"],
        rtol=1e-3,
        atol=1e-2,
    ).all()


def check_no_duplicate_rows(df: pd.DataFrame) -> bool:
    """Valida que el dataset no tenga filas duplicadas.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset que se desea validar.

    Returns
    -------
    bool
        True si no existen filas duplicadas.
    """
    return df.duplicated().sum() == 0


def get_churn_schema() -> pa.DataFrameSchema:
    """Construye el schema de validación del dataset de churn.

    Returns
    -------
    DataFrameSchema
        Schema de Pandera con reglas de tipos, rangos y consistencia.
    """
    return pa.DataFrameSchema(
        {
            "Attrition_Flag": pa.Column(
                str,
                nullable=False,
                checks=pa.Check.isin(["Existing Customer", "Attrited Customer"]),
            ),
            "Customer_Age": pa.Column(
                int,
                nullable=False,
                checks=pa.Check.in_range(18, 100),
                coerce=True,
            ),
            "Gender": pa.Column(
                str,
                nullable=False,
                checks=pa.Check.isin(["M", "F"]),
            ),
            "Dependent_count": pa.Column(
                int,
                nullable=False,
                checks=pa.Check.in_range(0, 10),
                coerce=True,
            ),
            "Education_Level": pa.Column(str, nullable=False),
            "Marital_Status": pa.Column(str, nullable=False),
            "Income_Category": pa.Column(str, nullable=False),
            "Card_Category": pa.Column(str, nullable=False),
            "Months_on_book": pa.Column(
                int,
                nullable=False,
                checks=pa.Check.greater_than(0),
                coerce=True,
            ),
            "Total_Relationship_Count": pa.Column(
                int,
                nullable=False,
                checks=pa.Check.in_range(1, 20),
                coerce=True,
            ),
            "Months_Inactive_12_mon": pa.Column(
                int,
                nullable=False,
                checks=pa.Check.in_range(0, 12),
                coerce=True,
            ),
            "Contacts_Count_12_mon": pa.Column(
                int,
                nullable=False,
                checks=pa.Check.in_range(0, 20),
                coerce=True,
            ),
            "Credit_Limit": pa.Column(
                float,
                nullable=False,
                checks=pa.Check.greater_than(0),
                coerce=True,
            ),
            "Total_Revolving_Bal": pa.Column(
                float,
                nullable=False,
                checks=pa.Check.greater_than_or_equal_to(0),
                coerce=True,
            ),
            "Avg_Open_To_Buy": pa.Column(
                float,
                nullable=False,
                checks=pa.Check.greater_than_or_equal_to(0),
                coerce=True,
            ),
            "Total_Amt_Chng_Q4_Q1": pa.Column(
                float,
                nullable=False,
                checks=pa.Check.greater_than_or_equal_to(0),
                coerce=True,
            ),
            "Total_Trans_Amt": pa.Column(
                float,
                nullable=False,
                checks=pa.Check.greater_than(0),
                coerce=True,
            ),
            "Total_Trans_Ct": pa.Column(
                int,
                nullable=False,
                checks=pa.Check.greater_than(0),
                coerce=True,
            ),
            "Total_Ct_Chng_Q4_Q1": pa.Column(
                float,
                nullable=False,
                checks=pa.Check.greater_than_or_equal_to(0),
                coerce=True,
            ),
            "Avg_Utilization_Ratio": pa.Column(
                float,
                nullable=False,
                checks=pa.Check.in_range(0, 1),
                coerce=True,
            ),
            "Avg_Trans_Amt": pa.Column(
                float,
                nullable=False,
                checks=pa.Check.greater_than(0),
                coerce=True,
            ),
            "Products_Per_Month": pa.Column(
                float,
                nullable=False,
                checks=pa.Check.greater_than(0),
                coerce=True,
            ),
            "Contacts_Per_Inactive_Month": pa.Column(
                float,
                nullable=False,
                checks=pa.Check.greater_than_or_equal_to(0),
                coerce=True,
            ),
        },
        checks=[
            pa.Check(
                check_credit_limit_consistency,
                error=(
                    "Credit_Limit debe ser consistente con "
                    "Total_Revolving_Bal + Avg_Open_To_Buy."
                ),
            ),
            pa.Check(
                check_avg_trans_amt,
                error=(
                    "Avg_Trans_Amt debe ser consistente con "
                    "Total_Trans_Amt / Total_Trans_Ct."
                ),
            ),
            pa.Check(
                check_no_duplicate_rows,
                error="El dataset no debe contener filas duplicadas.",
            ),
        ],
        strict=True,
        coerce=True,
    )


def validate_churn_data(df: pd.DataFrame, *, lazy: bool = True) -> pd.DataFrame:
    """Valida el dataset de churn con el schema del proyecto.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset que se desea validar.
    lazy : bool, default=True
        Si es True, Pandera acumula todos los errores antes de fallar.

    Returns
    -------
    pd.DataFrame
        Dataset validado.

    Raises
    ------
    pandera.errors.SchemaError
        Si el dataset no cumple con las reglas del schema.
    """
    schema = get_churn_schema()
    return schema.validate(df, lazy=lazy)
