"""Funciones de limpieza y feature engineering para el dataset de churn."""

import pandas as pd

ID_COLUMNS = ["CLIENTNUM"]

NAIVE_BAYES_COLUMNS = [
    (
        "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_"
        "Contacts_Count_12_mon_Dependent_count_Education_Level_"
        "Months_Inactive_12_mon_1"
    ),
    (
        "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_"
        "Contacts_Count_12_mon_Dependent_count_Education_Level_"
        "Months_Inactive_12_mon_2"
    ),
]

TARGET_MAP = {
    "Existing Customer": 0,
    "Attrited Customer": 1,
}


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina columnas que no deben usarse para modelado.

    Se eliminan el identificador del cliente y las columnas de Naive Bayes
    incluidas en el dataset original, ya que no representan variables reales
    del cliente para construir un modelo propio.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset original o intermedio.

    Returns
    -------
    pd.DataFrame
        DataFrame sin columnas irrelevantes.
    """
    columns_to_drop = [
        column for column in ID_COLUMNS + NAIVE_BAYES_COLUMNS if column in df.columns
    ]

    return df.drop(columns=columns_to_drop).copy()


def add_churn_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea variables derivadas para enriquecer el modelo de churn.

    Las variables creadas resumen comportamiento transaccional, intensidad
    de relación con el banco y frecuencia de contacto ajustada por inactividad.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset limpio con las variables originales necesarias.

    Returns
    -------
    pd.DataFrame
        DataFrame con variables derivadas adicionales.
    """
    df_features = df.copy()

    df_features["Avg_Trans_Amt"] = (
        df_features["Total_Trans_Amt"] / df_features["Total_Trans_Ct"]
    )

    df_features["Products_Per_Month"] = (
        df_features["Total_Relationship_Count"] / df_features["Months_on_book"]
    )

    df_features["Contacts_Per_Inactive_Month"] = df_features[
        "Contacts_Count_12_mon"
    ] / (df_features["Months_Inactive_12_mon"] + 1)

    return df_features


def build_churn_base_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Construye el dataset base de churn a partir del dataset crudo.

    Este flujo aplica la limpieza mínima del dataset original y después crea
    las variables derivadas que serán usadas por los modelos.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset crudo de churn.

    Returns
    -------
    pd.DataFrame
        Dataset base listo para validación y modelado.
    """
    clean_df = drop_unnecessary_columns(df)
    return add_churn_features(clean_df)


def encode_target(target: pd.Series) -> pd.Series:
    """Codifica la variable objetivo de churn como variable binaria.

    Parameters
    ----------
    target : pd.Series
        Serie con etiquetas originales del target.

    Returns
    -------
    pd.Series
        Serie codificada donde 0 representa cliente existente y 1 representa
        cliente que abandonó.
    """
    return target.map(TARGET_MAP)
