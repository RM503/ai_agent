# Python REPL tool
import ast
import json
import io
from typing import Annotated

import pandas as pd
from langchain.tools import tool

from .dataset_registry import DATASET_REGISTRY

def _parse_inline_data(data: str) -> pd.DataFrame:
    text = data.strip()

    if not text:
        raise ValueError(f"No inline data provided")

    # Try JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        if isinstance(obj, dict):
            try:
                return pd.DataFrame(obj)
            except Exception as e:
                return pd.DataFrame([obj])
    except Exception:
        pass

    # Try Python literal (single quoted list of dicts)
    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        if isinstance(obj, dict):
            try:
                return pd.DataFrame(obj)
            except Exception as e:
                return pd.DataFrame([obj])
    except Exception:
        pass

    raise ValueError("Could not parse inline data as JSON, Python literal, or CSV.")

@tool
def python_repl(
        code: Annotated[str, "The Python code to execute"],
        dataset_key: Annotated[str, "Registry key from file loader"] = "",
        data: Annotated[str, "Raw JSON, CSV or text data from user. Used when no file is uploaded"] = ""
) -> str:
    """
    This is a Python tool that executes Python code on loaded
    dataframes.

    Args:
        code (str): The code to execute.
        dataset_key (str): The key of the dataframe to execute.
        data (str): If data was passed through directly from UI as JSON
    Returns:
        str: The result of the execution.
    """
    if dataset_key:
        df = DATASET_REGISTRY.get(dataset_key, None)
        if df is None:
            return f"Dataset '{dataset_key} not found'"

    elif data.strip():
        try:
            df = _parse_inline_data(data)
        except Exception as e:
            return f"Error reading file: {e}"

    else:
        df = None # pure computation, no dataframe needed

    local_vars = {"df": df}
    safe_globals = {"pd": pd}

    try:
        exec(code, safe_globals, local_vars)
        return str(local_vars.get("result", "Execution complete"))
    except Exception as e:
        return f"Execution error: {e}"