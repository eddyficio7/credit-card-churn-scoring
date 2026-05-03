"""Modelo de churn basado en una red neuronal MLP con PyTorch."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from churn_scoring.models.base import BaseChurnModel


class ChurnMLP(nn.Module):
    """Red neuronal MLP para clasificación binaria de churn.

    Parameters
    ----------
    input_dim : int
        Número de variables de entrada.
    hidden_dims : tuple[int, ...], default=(64, 32)
        Tamaños de las capas ocultas.
    dropout : float, default=0.2
        Probabilidad de dropout entre capas ocultas.
    output_dim : int, default=2
        Número de salidas del modelo. Para clasificación binaria con
        CrossEntropyLoss normalmente se usan 2 logits.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (64, 32),
        dropout: float = 0.2,
        output_dim: int = 2,
    ) -> None:
        """Inicializa la arquitectura MLP."""
        super().__init__()

        layers: list[nn.Module] = []
        previous_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Ejecuta el forward pass del modelo.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Tensor de entrada con shape ``(n_muestras, n_features)``.

        Returns
        -------
        torch.Tensor
            Logits producidos por la red.
        """
        return self.net(input_tensor)


class TorchChurnModel(BaseChurnModel):
    """Wrapper de inferencia para modelos de churn entrenados con PyTorch.

    Esta clase adapta un modelo PyTorch a la interfaz común del proyecto,
    permitiendo generar predicciones y probabilidades desde un DataFrame.
    """

    def __init__(
        self,
        model: nn.Module,
        feature_columns: list[str],
        *,
        device: str | torch.device = "cpu",
        preprocessor: Any | None = None,
    ) -> None:
        """Inicializa el wrapper de inferencia PyTorch.

        Parameters
        ----------
        model : nn.Module
            Modelo PyTorch entrenado o inicializado.
        feature_columns : list[str]
            Columnas que se usarán como entrada si no se proporciona
            un preprocesador.
        device : str | torch.device, default="cpu"
            Dispositivo en el que se ejecutará la inferencia.
        preprocessor : Any | None, default=None
            Transformador opcional con método ``transform``. Puede ser un
            ColumnTransformer o Pipeline de scikit-learn.
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.feature_columns = feature_columns
        self.preprocessor = preprocessor

    @classmethod
    def from_state_dict(
        cls,
        checkpoint_path: str | Path,
        *,
        input_dim: int,
        feature_columns: list[str],
        hidden_dims: tuple[int, ...] = (64, 32),
        dropout: float = 0.2,
        output_dim: int = 2,
        device: str | torch.device = "cpu",
        preprocessor: Any | None = None,
    ) -> "TorchChurnModel":
        """Construye un wrapper a partir de un state_dict guardado.

        Parameters
        ----------
        checkpoint_path : str | Path
            Ruta del archivo ``.pt`` con los pesos del modelo.
        input_dim : int
            Número de variables de entrada del modelo.
        feature_columns : list[str]
            Columnas que se usarán como entrada.
        hidden_dims : tuple[int, ...], default=(64, 32)
            Tamaños de capas ocultas.
        dropout : float, default=0.2
            Probabilidad de dropout usada en la arquitectura.
        output_dim : int, default=2
            Número de salidas del modelo.
        device : str | torch.device, default="cpu"
            Dispositivo para cargar el modelo.
        preprocessor : Any | None, default=None
            Preprocesador opcional para transformar el DataFrame.

        Returns
        -------
        TorchChurnModel
            Wrapper listo para inferencia.

        Raises
        ------
        FileNotFoundError
            Si el archivo del checkpoint no existe.
        """
        path = Path(checkpoint_path)

        if not path.exists():
            raise FileNotFoundError(f"No se encontró el checkpoint: {path}")

        model = ChurnMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            output_dim=output_dim,
        )
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)

        return cls(
            model=model,
            feature_columns=feature_columns,
            device=device,
            preprocessor=preprocessor,
        )

    def _prepare_input(self, input_data: pd.DataFrame) -> torch.Tensor:
        """Convierte un DataFrame en tensor para inferencia.

        Parameters
        ----------
        input_data : pd.DataFrame
            DataFrame con variables explicativas.

        Returns
        -------
        torch.Tensor
            Tensor de entrada en ``float32``.
        """
        if self.preprocessor is not None:
            transformed_data = self.preprocessor.transform(input_data)
            input_array = np.asarray(transformed_data, dtype=np.float32)
        else:
            input_array = input_data[self.feature_columns].to_numpy(
                dtype=np.float32,
            )

        return torch.tensor(input_array, dtype=torch.float32, device=self.device)

    @torch.no_grad()
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
        input_tensor = self._prepare_input(input_data)
        logits = self.model(input_tensor)

        if logits.ndim == 1 or logits.shape[1] == 1:
            probabilities = torch.sigmoid(logits.reshape(-1))
        else:
            probabilities = torch.softmax(logits, dim=1)[:, 1]

        return pd.Series(
            probabilities.cpu().numpy(),
            index=input_data.index,
            name="churn_probability",
        )

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
        probabilities = self.predict_proba(input_data)
        predictions = (probabilities >= 0.5).astype(int)

        return pd.Series(
            predictions,
            index=input_data.index,
            name="prediction",
        )
