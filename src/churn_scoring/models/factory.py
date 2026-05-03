"""Factory para construir modelos de churn."""

from pathlib import Path
from typing import Literal

from churn_scoring.models.base import BaseChurnModel
from churn_scoring.models.sklearn_model import SklearnChurnModel
from churn_scoring.models.torch_model import TorchChurnModel

ModelType = Literal["sklearn", "torch"]


def create_churn_model(
    model_type: ModelType,
    model_path: str | Path,
    *,
    input_dim: int | None = None,
    feature_columns: list[str] | None = None,
    hidden_dims: tuple[int, ...] = (64, 32),
    dropout: float = 0.2,
    output_dim: int = 2,
    device: str = "cpu",
) -> BaseChurnModel:
    """Crea un modelo de churn según el tipo solicitado.

    Esta función implementa el patrón Factory para centralizar la creación
    de modelos. Permite que la aplicación use una interfaz común sin conocer
    los detalles internos de scikit-learn o PyTorch.

    Parameters
    ----------
    model_type : {"sklearn", "torch"}
        Tipo de modelo que se desea crear.
    model_path : str | Path
        Ruta del artefacto serializado del modelo.
    input_dim : int | None, default=None
        Número de variables de entrada. Es obligatorio para modelos PyTorch.
    feature_columns : list[str] | None, default=None
        Columnas de entrada. Es obligatorio para modelos PyTorch cuando no se
        proporciona un preprocesador.
    hidden_dims : tuple[int, ...], default=(64, 32)
        Tamaños de capas ocultas del modelo PyTorch.
    dropout : float, default=0.2
        Probabilidad de dropout del modelo PyTorch.
    output_dim : int, default=2
        Número de salidas del modelo PyTorch.
    device : str, default="cpu"
        Dispositivo usado para inferencia en PyTorch.

    Returns
    -------
    BaseChurnModel
        Modelo construido con la interfaz común del proyecto.

    Raises
    ------
    ValueError
        Si el tipo de modelo no está soportado o faltan parámetros requeridos.
    """
    if model_type == "sklearn":
        return SklearnChurnModel(model_path)

    if model_type == "torch":
        if input_dim is None:
            raise ValueError("input_dim es obligatorio para modelos PyTorch.")

        if feature_columns is None:
            raise ValueError("feature_columns es obligatorio para modelos PyTorch.")

        return TorchChurnModel.from_state_dict(
            model_path,
            input_dim=input_dim,
            feature_columns=feature_columns,
            hidden_dims=hidden_dims,
            dropout=dropout,
            output_dim=output_dim,
            device=device,
        )

    raise ValueError(f"Tipo de modelo no soportado: {model_type}")
