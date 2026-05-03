"""Funciones para cargar y guardar datasets del proyecto."""

from pathlib import Path

import pandas as pd


def load_csv_dataset(path: str | Path) -> pd.DataFrame:
    """Carga un dataset en formato CSV.

    Parameters
    ----------
    path : str | Path
        Ruta del archivo CSV que se desea cargar.

    Returns
    -------
    pd.DataFrame
        Dataset cargado como DataFrame de pandas.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe en la ruta indicada.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")

    return pd.read_csv(file_path)


def load_parquet_dataset(path: str | Path) -> pd.DataFrame:
    """Carga un dataset en formato Parquet.

    Parameters
    ----------
    path : str | Path
        Ruta del archivo Parquet que se desea cargar.

    Returns
    -------
    pd.DataFrame
        Dataset cargado como DataFrame de pandas.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe en la ruta indicada.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")

    return pd.read_parquet(file_path)


def save_parquet_dataset(
    df: pd.DataFrame,
    path: str | Path,
    *,
    index: bool = False,
) -> None:
    """Guarda un DataFrame en formato Parquet.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame que se desea guardar.
    path : str | Path
        Ruta de salida del archivo Parquet.
    index : bool, default=False
        Indica si se debe guardar el índice del DataFrame.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(file_path, index=index, engine="pyarrow", compression="snappy")
