"""Entrena y serializa un modelo MLP de PyTorch para churn."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from churn_scoring.config import get_settings
from churn_scoring.data.loader import load_parquet_dataset
from churn_scoring.features.engineering import encode_target
from churn_scoring.models.torch_model import ChurnMLP


@dataclass(frozen=True)
class TorchTrainingConfig:
    """Configuración de entrenamiento para el modelo PyTorch."""

    random_state: int = 42
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    n_epochs: int = 50
    patience: int = 8
    hidden_dims: tuple[int, ...] = (64, 32)
    dropout: float = 0.2


class ChurnDataset(Dataset):
    """Dataset personalizado para entrenamiento del MLP de churn."""

    def __init__(self, features: np.ndarray, target: np.ndarray) -> None:
        """Inicializa el dataset a partir de arreglos NumPy.

        Parameters
        ----------
        features : np.ndarray
            Matriz de variables explicativas.
        target : np.ndarray
            Vector con la variable objetivo.
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.long)

    def __len__(self) -> int:
        """Devuelve el número de observaciones del dataset."""
        return len(self.target)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Devuelve una observación y su etiqueta."""
        return self.features[index], self.target[index]


def set_seed(seed: int) -> None:
    """Fija semillas para mejorar reproducibilidad."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_preprocessor(x_train: pd.DataFrame) -> ColumnTransformer:
    """Construye el preprocesador para el modelo PyTorch.

    Parameters
    ----------
    x_train : pd.DataFrame
        Variables de entrenamiento.

    Returns
    -------
    ColumnTransformer
        Preprocesador ajustable para variables numéricas y categóricas.
    """
    numeric_features = x_train.select_dtypes(
        include=["int64", "float64", "int32", "float32"],
    ).columns.tolist()

    categorical_features = x_train.select_dtypes(
        include=["object", "category", "string"],
    ).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ],
        remainder="drop",
    )


def make_data_loader(
    features: np.ndarray,
    target: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Construye un DataLoader para entrenamiento o evaluación."""
    dataset = ChurnDataset(features, target)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Entrena el modelo durante una época.

    Returns
    -------
    float
        Pérdida promedio de entrenamiento.
    """
    model.train()

    running_loss = 0.0
    n_samples = 0

    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        loss.backward()
        optimizer.step()

        batch_size = x_batch.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

    return running_loss / n_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evalúa el modelo en un conjunto de datos.

    Returns
    -------
    dict[str, float]
        Diccionario con loss, F1 y ROC AUC.
    """
    model.eval()

    running_loss = 0.0
    n_samples = 0
    all_targets = []
    all_probabilities = []

    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        probabilities = torch.softmax(logits, dim=1)[:, 1]

        batch_size = x_batch.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

        all_targets.extend(y_batch.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

    targets = np.array(all_targets)
    probabilities_array = np.array(all_probabilities)
    predictions = (probabilities_array >= 0.5).astype(int)

    return {
        "loss": running_loss / n_samples,
        "f1": float(f1_score(targets, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(targets, probabilities_array)),
    }


def save_joblib_artifact(artifact: Any, path: str | Path) -> None:
    """Guarda un artefacto con joblib usando context manager."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as file:
        joblib.dump(artifact, file)


def main() -> None:
    """Ejecuta el entrenamiento completo del MLP de churn."""
    settings = get_settings()
    config = TorchTrainingConfig(random_state=settings.random_state)

    set_seed(config.random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = load_parquet_dataset(settings.data_validated_path)

    x = df.drop(columns=[settings.target_column])
    y = encode_target(df[settings.target_column]).to_numpy()

    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.40,
        stratify=y,
        random_state=config.random_state,
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=config.random_state,
    )

    preprocessor = build_preprocessor(x_train)

    x_train_processed = preprocessor.fit_transform(x_train)
    x_val_processed = preprocessor.transform(x_val)
    x_test_processed = preprocessor.transform(x_test)

    save_joblib_artifact(preprocessor, settings.pytorch_preprocessor_path)

    train_loader = make_data_loader(
        x_train_processed,
        y_train,
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = make_data_loader(
        x_val_processed,
        y_val,
        batch_size=config.batch_size,
        shuffle=False,
    )
    test_loader = make_data_loader(
        x_test_processed,
        y_test,
        batch_size=config.batch_size,
        shuffle=False,
    )

    input_dim = x_train_processed.shape[1]

    model = ChurnMLP(
        input_dim=input_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
        output_dim=2,
    ).to(device)

    class_counts = np.bincount(y_train)
    class_weights = class_counts.sum() / (len(class_counts) * class_counts)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_val_f1 = -np.inf
    best_state_dict = None
    patience_counter = 0

    for epoch in range(1, config.n_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_metrics = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"val_auc={val_metrics['roc_auc']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            print("Early stopping activado.")
            break

    if best_state_dict is None:
        raise RuntimeError("No se pudo entrenar el modelo PyTorch.")

    model.load_state_dict(best_state_dict)

    settings.pytorch_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state_dict, settings.pytorch_model_path)

    test_metrics = evaluate(model, test_loader, criterion, device)

    print("\nArtefactos guardados:")
    print(f"Modelo PyTorch: {settings.pytorch_model_path}")
    print(f"Preprocesador:  {settings.pytorch_preprocessor_path}")
    print("\nMétricas en test:")
    print(f"F1:      {test_metrics['f1']:.4f}")
    print(f"ROC AUC: {test_metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
