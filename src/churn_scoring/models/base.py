"""Interfaces y estructuras base para modelos de churn."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ChurnPrediction:
    """Representa una predicción individual de churn.

    Attributes
    ----------
    prediction : int
        Clase predicha por el modelo. El valor 1 representa churn y el valor 0
        representa cliente existente.
    churn_probability : float
        Probabilidad estimada de churn.
    risk_level : str
        Nivel de riesgo asignado a partir de la probabilidad de churn.
    recommendation : str
        Recomendación de negocio asociada al nivel de riesgo.
    """

    prediction: int
    churn_probability: float
    risk_level: str
    recommendation: str


class BaseChurnModel(ABC):
    """Interfaz base para modelos de predicción de churn.

    Esta clase define el contrato mínimo que deben cumplir los modelos del
    proyecto, sin importar si fueron implementados con scikit-learn o PyTorch.
    """

    @abstractmethod
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

    @abstractmethod
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
        """
