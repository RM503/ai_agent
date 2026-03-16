# File loader tool

from pathlib import Path
from typing import Annotated, Callable

import pandas as pd
from langchain.tools import tool

from .dataset_registry import DATASET_REGISTRY

LoaderType = Callable[[Path | str], pd.DataFrame]

def csv_loader(file_path: Path | str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def excel_loader(file_path: Path | str) -> pd.DataFrame:
    return pd.read_excel(file_path)

_LOADER_REGISTRY: dict[str, LoaderType] = {
    ".csv": csv_loader,
    ".xlsx": excel_loader,
    ".xls": excel_loader,
}

@tool
def file_loader(file_path: Annotated[str, "Absolute or relative path to the file"]) -> str:
    """
    This function is used to load data from a given file.

    Args:
        file_path (Path): the file path
    Returns:
        Text sumary containing dataset key, shape, columns and a small
        preview so that the agent can decide what to do next.
    """

    path = Path(file_path)
    if not path.exists():
        return f"Error: '{file_path}' does not exist. Verify upload and retry."

    extension = path.suffix
    loader = _LOADER_REGISTRY.get(extension, None)

    if loader is None:
        supported = ", ".join(_LOADER_REGISTRY)
        return f"Error: unsupported file type '{extension}. Supported types are {supported}"

    try:
        df = loader(path)
    except Exception as e:
        return f"Error loading file: {e}"

    # Use the datasets name as the key for registry (sans extension)
    dataset_key = path.name
    preview = df.head(5).to_string()
    DATASET_REGISTRY[dataset_key] = df

    return (
        f"Data successfully loaded.\n"
        f"Dataset key: {dataset_key}\n"
        f"Shape: {df.shape[0]} rows and {df.shape[1]} columns \n"
        f"Columns: {', '.join(map(str, df.columns.tolist()))}\n"
        f"Preview: \n{preview}\n"
        f"Use dataset key '{dataset_key}' for follow-up analysis."
    )