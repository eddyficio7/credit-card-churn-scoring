"""Pruebas para funciones de evaluación de modelos."""

import numpy as np
import pytest

from churn_scoring.evaluation.metrics import (
    ClassificationMetrics,
    calculate_binary_classification_metrics,
)


def test_calculate_binary_classification_metrics_without_probabilities() -> None:
    """Valida métricas binarias cuando no se reciben probabilidades."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])

    metrics = calculate_binary_classification_metrics(y_true, y_pred)

    assert isinstance(metrics, ClassificationMetrics)
    assert metrics.accuracy == pytest.approx(0.75)
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(0.8)
    assert metrics.roc_auc is None


def test_calculate_binary_classification_metrics_with_probabilities() -> None:
    """Valida métricas binarias cuando se reciben probabilidades."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_proba = np.array([0.10, 0.70, 0.80, 0.90])

    metrics = calculate_binary_classification_metrics(y_true, y_pred, y_proba)

    assert metrics.accuracy == pytest.approx(0.75)
    assert metrics.roc_auc == pytest.approx(1.0)


def test_classification_metrics_to_dict() -> None:
    """Valida la conversión de métricas a diccionario."""
    metrics = ClassificationMetrics(
        accuracy=0.9,
        precision=0.8,
        recall=0.7,
        f1=0.75,
        roc_auc=0.95,
    )

    result = metrics.to_dict()

    assert result == {
        "accuracy": 0.9,
        "precision": 0.8,
        "recall": 0.7,
        "f1": 0.75,
        "roc_auc": 0.95,
    }
