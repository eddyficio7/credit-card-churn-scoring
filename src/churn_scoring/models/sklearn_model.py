"""Modelo de churn basado en un pipeline serializado de scikit-learn."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from churn_scoring.models.base import BaseChurnModel, ChurnPrediction


class SklearnChurnModel(BaseChurnModel):
    """Wrapper para modelos de churn entrenados con scikit-learn.

    Esta clase carga un pipeline serializado con joblib y expone métodos
    estándar para generar predicciones y probabilidades de churn.
    """

    def __init__(self, model_path: str | Path) -> None:
        """Inicializa el modelo cargando el artefacto serializado.

        Parameters
        ----------
        model_path : str | Path
            Ruta del archivo joblib con el pipeline entrenado.
        """
        self.model_path = Path(model_path)
        self.model = self._load_model(self.model_path)

    @staticmethod
    def _load_model(model_path: Path) -> Any:
        """Carga un modelo serializado con joblib.

        Parameters
        ----------
        model_path : Path
            Ruta del archivo del modelo.

        Returns
        -------
        Any
            Modelo o pipeline cargado.

        Raises
        ------
        FileNotFoundError
            Si el archivo del modelo no existe.
        """
        if not model_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo: {model_path}")

        with model_path.open("rb") as file:
            return joblib.load(file)

    def predict(self, input_data: pd.DataFrame) -> pd.Series:
        """Genera predicciones de clase para los clientes recibidos.

        Parameters
        ----------
        input_data : pd.DataFrame
            Variables explicativas de los clientes.

        Returns
        -------
        pd.Series
            Serie con las clases predichas.
        """
        predictions = self.model.predict(input_data)
        return pd.Series(predictions, name="prediction")

    def predict_proba(self, input_data: pd.DataFrame) -> pd.Series:
        """Genera probabilidades de churn para los clientes recibidos.

        Parameters
        ----------
        input_data : pd.DataFrame
            Variables explicativas de los clientes.

        Returns
        -------
        pd.Series
            Serie con la probabilidad estimada de churn.

        Raises
        ------
        AttributeError
            Si el modelo cargado no implementa predict_proba.
        """
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("El modelo cargado no implementa predict_proba.")

        probabilities = self.model.predict_proba(input_data)

        return pd.Series(probabilities[:, 1], name="churn_probability")

    def predict_one(self, input_data: pd.DataFrame) -> ChurnPrediction:
        """Genera una predicción enriquecida para un solo cliente.

        Parameters
        ----------
        input_data : pd.DataFrame
            DataFrame con una sola fila que representa al cliente.

        Returns
        -------
        ChurnPrediction
            Predicción con probabilidad, nivel de riesgo y recomendación.

        Raises
        ------
        ValueError
            Si el DataFrame no contiene exactamente una fila.
        """
        if len(input_data) != 1:
            raise ValueError(
                "predict_one espera un DataFrame con exactamente una fila."
            )

        prediction = int(self.predict(input_data).iloc[0])
        churn_probability = float(self.predict_proba(input_data).iloc[0])

        risk_level = assign_risk_level(churn_probability)
        recommendation = assign_recommendation(risk_level)

        return ChurnPrediction(
            prediction=prediction,
            churn_probability=churn_probability,
            risk_level=risk_level,
            recommendation=recommendation,
        )


def assign_risk_level(churn_probability: float) -> str:
    """Asigna un nivel de riesgo a partir de la probabilidad de churn.

    Parameters
    ----------
    churn_probability : float
        Probabilidad estimada de churn.

    Returns
    -------
    str
        Nivel de riesgo: Bajo, Medio o Alto.
    """
    if churn_probability < 0.30:
        return "Bajo"

    if churn_probability < 0.60:
        return "Medio"

    return "Alto"


def assign_recommendation(risk_level: str) -> str:
    """Asigna una recomendación de negocio según el nivel de riesgo.

    Parameters
    ----------
    risk_level : str
        Nivel de riesgo estimado.

    Returns
    -------
    str
        Recomendación de retención.
    """
    recommendations = {
        "Bajo": "Mantener estrategia actual de relación con el cliente.",
        "Medio": "Monitorear comportamiento y considerar una oferta preventiva.",
        "Alto": "Priorizar acción de retención personalizada.",
    }

    return recommendations[risk_level]
