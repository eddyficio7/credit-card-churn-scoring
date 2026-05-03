"""Cálculo de métricas de evaluación para modelos de clasificación."""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    """Representa métricas principales de clasificación binaria.

    Attributes
    ----------
    accuracy : float
        Proporción de predicciones correctas.
    precision : float
        Precisión de la clase positiva.
    recall : float
        Sensibilidad o cobertura de la clase positiva.
    f1 : float
        Media armónica entre precision y recall.
    roc_auc : float | None
        Área bajo la curva ROC. Puede ser None si no se reciben
        probabilidades o scores continuos.
    """

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convierte las métricas a diccionario.

        Returns
        -------
        dict[str, Any]
            Diccionario con las métricas calculadas.
        """
        return asdict(self)


def calculate_binary_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> ClassificationMetrics:
    """Calcula métricas principales para clasificación binaria.

    Parameters
    ----------
    y_true : np.ndarray
        Etiquetas reales.
    y_pred : np.ndarray
        Clases predichas por el modelo.
    y_proba : np.ndarray | None, default=None
        Probabilidades o scores de la clase positiva.

    Returns
    -------
    ClassificationMetrics
        Métricas calculadas para el modelo.
    """
    roc_auc = None

    if y_proba is not None:
        roc_auc = float(roc_auc_score(y_true, y_proba))

    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=roc_auc,
    )
