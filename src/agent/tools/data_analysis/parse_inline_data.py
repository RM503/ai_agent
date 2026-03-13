import ast
import json
from uuid import uuid4

import pandas as pd

from .dataset_registry import DATASET_REGISTRY

def parse_inline_data(data: str) -> pd.DataFrame:
    """
    This function parses inline data from UI and returns a pandas dataframe.
    """
    text = data.strip()

    if not text:
        raise ValueError(f"No inline data provided")

    # Try JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        if isinstance(obj, dict):
            # Convert to dataframe if all keys of dictionary are lists
            return pd.DataFrame(obj) if all(isinstance(v, list) for v in obj.values()) else [obj]
    except Exception as e:
        pass

    # Try Python literal (single quoted list of dicts)
    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        if isinstance(obj, dict):
            return pd.DataFrame(obj) if all(isinstance(v, list) for v in obj.values()) else [obj]
    except Exception:
        pass

    raise ValueError("Could not parse inline data as JSON, Python literal, or CSV.")

def register_inline_dataset(text: str) -> str:
    df = parse_inline_data(text)
    dataset_key = f"inline_{str(uuid4().hex)}"
    DATASET_REGISTRY[dataset_key] = df
    return dataset_key