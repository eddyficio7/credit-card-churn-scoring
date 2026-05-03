"""Pruebas para modelos de churn implementados con PyTorch."""

from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn

from churn_scoring.models.torch_model import ChurnMLP, TorchChurnModel


def make_sample_input() -> pd.DataFrame:
    """Construye un DataFrame mínimo para probar inferencia."""
    return pd.DataFrame(
        {
            "feature_1": [1.0, 2.0],
            "feature_2": [0.5, 1.5],
        }
    )


def test_churn_mlp_forward_returns_expected_shape() -> None:
    """Valida que el MLP devuelva logits con la dimensión esperada."""
    model = ChurnMLP(input_dim=2, hidden_dims=(4,), dropout=0.0, output_dim=2)
    input_tensor = torch.randn(3, 2)

    logits = model(input_tensor)

    assert logits.shape == (3, 2)


def test_torch_churn_model_predict_proba_returns_series() -> None:
    """Valida que predict_proba devuelva probabilidades como Serie."""
    torch.manual_seed(42)
    model = ChurnMLP(input_dim=2, hidden_dims=(4,), dropout=0.0, output_dim=2)
    wrapper = TorchChurnModel(
        model=model,
        feature_columns=["feature_1", "feature_2"],
    )

    probabilities = wrapper.predict_proba(make_sample_input())

    assert isinstance(probabilities, pd.Series)
    assert probabilities.name == "churn_probability"
    assert len(probabilities) == 2
    assert probabilities.between(0, 1).all()


def test_torch_churn_model_predict_returns_binary_series() -> None:
    """Valida que predict devuelva clases binarias."""
    torch.manual_seed(42)
    model = ChurnMLP(input_dim=2, hidden_dims=(4,), dropout=0.0, output_dim=2)
    wrapper = TorchChurnModel(
        model=model,
        feature_columns=["feature_1", "feature_2"],
    )

    predictions = wrapper.predict(make_sample_input())

    assert isinstance(predictions, pd.Series)
    assert predictions.name == "prediction"
    assert set(predictions.unique()).issubset({0, 1})


def test_torch_churn_model_uses_eval_mode() -> None:
    """Valida que el wrapper coloque el modelo en modo evaluación."""
    model = ChurnMLP(input_dim=2, hidden_dims=(4,), dropout=0.2, output_dim=2)

    wrapper = TorchChurnModel(
        model=model,
        feature_columns=["feature_1", "feature_2"],
    )

    assert wrapper.model.training is False


def test_torch_churn_model_from_state_dict_loads_checkpoint(
    tmp_path: Path,
) -> None:
    """Valida que el wrapper pueda cargarse desde un state_dict."""
    checkpoint_path = tmp_path / "model.pt"
    model = ChurnMLP(input_dim=2, hidden_dims=(4,), dropout=0.0, output_dim=2)
    torch.save(model.state_dict(), checkpoint_path)

    wrapper = TorchChurnModel.from_state_dict(
        checkpoint_path,
        input_dim=2,
        hidden_dims=(4,),
        dropout=0.0,
        output_dim=2,
        feature_columns=["feature_1", "feature_2"],
    )

    probabilities = wrapper.predict_proba(make_sample_input())

    assert len(probabilities) == 2


def test_torch_churn_model_from_state_dict_raises_file_not_found(
    tmp_path: Path,
) -> None:
    """Valida que se lance error si no existe el checkpoint."""
    checkpoint_path = tmp_path / "missing_model.pt"

    with pytest.raises(FileNotFoundError):
        TorchChurnModel.from_state_dict(
            checkpoint_path,
            input_dim=2,
            feature_columns=["feature_1", "feature_2"],
        )


def test_torch_churn_model_supports_binary_single_logit_output() -> None:
    """Valida inferencia cuando el modelo devuelve un solo logit."""
    model = nn.Linear(2, 1)
    wrapper = TorchChurnModel(
        model=model,
        feature_columns=["feature_1", "feature_2"],
    )

    probabilities = wrapper.predict_proba(make_sample_input())

    assert len(probabilities) == 2
    assert probabilities.between(0, 1).all()
