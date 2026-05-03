"""Pruebas para el wrapper de modelos scikit-learn."""

from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from churn_scoring.models.base import ChurnPrediction
from churn_scoring.models.sklearn_model import (
    SklearnChurnModel,
    assign_recommendation,
    assign_risk_level,
)


def make_sample_input() -> pd.DataFrame:
    """Construye un DataFrame mínimo para probar inferencia."""
    return pd.DataFrame(
        {
            "feature_1": [1.0, 2.0],
            "feature_2": [0.5, 1.5],
        }
    )


def make_fitted_dummy_model() -> DummyClassifier:
    """Construye un modelo dummy entrenado para pruebas."""
    x_train = pd.DataFrame(
        {
            "feature_1": [1.0, 2.0, 3.0, 4.0],
            "feature_2": [0.5, 1.5, 2.5, 3.5],
        }
    )
    y_train = [0, 1, 0, 1]

    model = DummyClassifier(strategy="prior")
    model.fit(x_train, y_train)

    return model


def test_sklearn_churn_model_loads_joblib_model(tmp_path: Path) -> None:
    """Valida que el wrapper cargue un modelo joblib existente."""
    model_path = tmp_path / "dummy_model.joblib"
    dummy_model = make_fitted_dummy_model()
    joblib.dump(dummy_model, model_path)

    model = SklearnChurnModel(model_path)

    assert model.model_path == model_path
    assert model.model is not None


def test_sklearn_churn_model_raises_when_model_does_not_exist(
    tmp_path: Path,
) -> None:
    """Valida que se lance error si el modelo no existe."""
    model_path = tmp_path / "missing_model.joblib"

    with pytest.raises(FileNotFoundError):
        SklearnChurnModel(model_path)


def test_predict_returns_series(tmp_path: Path) -> None:
    """Valida que predict devuelva una Serie con predicciones."""
    model_path = tmp_path / "dummy_model.joblib"
    dummy_model = make_fitted_dummy_model()
    joblib.dump(dummy_model, model_path)

    model = SklearnChurnModel(model_path)
    predictions = model.predict(make_sample_input())

    assert isinstance(predictions, pd.Series)
    assert predictions.name == "prediction"
    assert len(predictions) == 2


def test_predict_proba_returns_churn_probability_series(tmp_path: Path) -> None:
    """Valida que predict_proba devuelva probabilidades de churn."""
    model_path = tmp_path / "dummy_model.joblib"
    dummy_model = make_fitted_dummy_model()
    joblib.dump(dummy_model, model_path)

    model = SklearnChurnModel(model_path)
    probabilities = model.predict_proba(make_sample_input())

    assert isinstance(probabilities, pd.Series)
    assert probabilities.name == "churn_probability"
    assert len(probabilities) == 2
    assert probabilities.between(0, 1).all()


def test_predict_one_returns_churn_prediction(tmp_path: Path) -> None:
    """Valida que predict_one devuelva una predicción enriquecida."""
    model_path = tmp_path / "dummy_model.joblib"
    dummy_model = make_fitted_dummy_model()
    joblib.dump(dummy_model, model_path)

    model = SklearnChurnModel(model_path)
    result = model.predict_one(make_sample_input().iloc[[0]])

    assert isinstance(result, ChurnPrediction)
    assert result.risk_level in {"Bajo", "Medio", "Alto"}
    assert isinstance(result.recommendation, str)


def test_predict_one_raises_with_multiple_rows(tmp_path: Path) -> None:
    """Valida que predict_one rechace DataFrames con más de una fila."""
    model_path = tmp_path / "dummy_model.joblib"
    dummy_model = make_fitted_dummy_model()
    joblib.dump(dummy_model, model_path)

    model = SklearnChurnModel(model_path)

    with pytest.raises(ValueError):
        model.predict_one(make_sample_input())


@pytest.mark.parametrize(
    ("probability", "expected_level"),
    [
        (0.10, "Bajo"),
        (0.30, "Medio"),
        (0.59, "Medio"),
        (0.60, "Alto"),
        (0.90, "Alto"),
    ],
)
def test_assign_risk_level(probability: float, expected_level: str) -> None:
    """Valida la asignación de nivel de riesgo."""
    assert assign_risk_level(probability) == expected_level


@pytest.mark.parametrize(
    "risk_level",
    ["Bajo", "Medio", "Alto"],
)
def test_assign_recommendation_returns_text(risk_level: str) -> None:
    """Valida que cada nivel de riesgo tenga recomendación."""
    recommendation = assign_recommendation(risk_level)

    assert isinstance(recommendation, str)
    assert len(recommendation) > 0
