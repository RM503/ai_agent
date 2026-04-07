# File loader tool

import json
from pathlib import Path
from typing import Annotated, Any, Callable

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
        str: JSON result.

        Success shape:
            {
                "status": "ok",
                "dataset_key": str,
                "shape": [int, int],
                "columns": list[str],
                "preview_rows": list[dict[str, Any]]
            }

        Error shape:
            {
                "status": "error",
                "error": str
            }
    """

    path = Path(file_path)
    if not path.exists():
        return json.dumps({
            "status": "error",
            "error": f"File at '{file_path}' not found. Verify upload."
        })

    extension = path.suffix
    loader = _LOADER_REGISTRY.get(extension, None)

    if loader is None:
        supported = ", ".join(_LOADER_REGISTRY)
        return json.dumps({
            "status": "error",
            "error": f"File extension '{extension}' not supported. Supported types are {supported}"
        })

    try:
        df = loader(path)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": f"Error loading file: {e}"
        })

    # Use the datasets name as the key for registry (sans extension)
    dataset_key = path.name
    preview_rows = df.head(5).to_dict(orient="records")
    DATASET_REGISTRY[dataset_key] = df

    return json.dumps({
        "status": "ok",
        "dataset_key": dataset_key,
        "shape": [df.shape[0], df.shape[1]],
        "columns": list(map(str, df.columns.tolist())),
        "preview_rows": preview_rows
    })