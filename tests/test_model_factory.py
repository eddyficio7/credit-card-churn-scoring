"""Pruebas para la factory de modelos de churn."""

from pathlib import Path
from typing import cast

import joblib
import pytest
import torch
from sklearn.dummy import DummyClassifier

from churn_scoring.models.base import BaseChurnModel
from churn_scoring.models.factory import ModelType, create_churn_model
from churn_scoring.models.sklearn_model import SklearnChurnModel
from churn_scoring.models.torch_model import ChurnMLP, TorchChurnModel


def make_sklearn_model_path(tmp_path: Path) -> Path:
    """Construye y guarda un modelo dummy de scikit-learn."""
    model_path = tmp_path / "dummy_model.joblib"

    x_train = [[1.0, 0.5], [2.0, 1.5], [3.0, 2.5], [4.0, 3.5]]
    y_train = [0, 1, 0, 1]

    model = DummyClassifier(strategy="prior")
    model.fit(x_train, y_train)

    joblib.dump(model, model_path)

    return model_path


def make_torch_checkpoint_path(tmp_path: Path) -> Path:
    """Construye y guarda un checkpoint dummy de PyTorch."""
    checkpoint_path = tmp_path / "model.pt"
    model = ChurnMLP(input_dim=2, hidden_dims=(4,), dropout=0.0, output_dim=2)

    torch.save(model.state_dict(), checkpoint_path)

    return checkpoint_path


def test_create_churn_model_returns_sklearn_model(tmp_path: Path) -> None:
    """Valida que la factory construya un modelo scikit-learn."""
    model_path = make_sklearn_model_path(tmp_path)

    model = create_churn_model("sklearn", model_path)

    assert isinstance(model, BaseChurnModel)
    assert isinstance(model, SklearnChurnModel)


def test_create_churn_model_returns_torch_model(tmp_path: Path) -> None:
    """Valida que la factory construya un modelo PyTorch."""
    checkpoint_path = make_torch_checkpoint_path(tmp_path)

    model = create_churn_model(
        "torch",
        checkpoint_path,
        input_dim=2,
        feature_columns=["feature_1", "feature_2"],
        hidden_dims=(4,),
        dropout=0.0,
        output_dim=2,
    )

    assert isinstance(model, BaseChurnModel)
    assert isinstance(model, TorchChurnModel)


def test_create_churn_model_requires_input_dim_for_torch(
    tmp_path: Path,
) -> None:
    """Valida que PyTorch requiera input_dim."""
    checkpoint_path = make_torch_checkpoint_path(tmp_path)

    with pytest.raises(ValueError, match="input_dim"):
        create_churn_model(
            "torch",
            checkpoint_path,
            feature_columns=["feature_1", "feature_2"],
        )


def test_create_churn_model_requires_feature_columns_for_torch(
    tmp_path: Path,
) -> None:
    """Valida que PyTorch requiera feature_columns."""
    checkpoint_path = make_torch_checkpoint_path(tmp_path)

    with pytest.raises(ValueError, match="feature_columns"):
        create_churn_model(
            "torch",
            checkpoint_path,
            input_dim=2,
        )


def test_create_churn_model_rejects_unsupported_model_type(
    tmp_path: Path,
) -> None:
    """Valida que se rechacen tipos de modelo no soportados."""
    model_path = tmp_path / "model.joblib"
    unsupported_type = cast(ModelType, "xgboost")

    with pytest.raises(ValueError, match="Tipo de modelo no soportado"):
        create_churn_model(unsupported_type, model_path)
