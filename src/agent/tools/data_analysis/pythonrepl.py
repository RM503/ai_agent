# Python REPL tool
import ast
import json
import re
from typing import Annotated

import pandas as pd
from langchain.tools import tool

from .dataset_registry import DATASET_REGISTRY
from .parse_inline_data import register_inline_dataset

def _violates_code_rules(code: str) -> str | None:
    checks = [
        (r"\bdf\s*=", "Do not assign to `df`. The dataframe is already provided."),
        (r"\bdata\s*=\s*\[", "Do not inline dataset rows inside the code."),
        (r"\bdata\s*=\s*\{", "Do not inline dataset values inside the code."),
        (r"json\.loads\s*\(", "Do not parse JSON inside the code."),
        (r"ast\.literal_eval\s*\(", "Do not parse literals inside the code."),
        (r"pd\.read_", "Do not load files inside the code."),
        (r"pd\.DataFrame\s*\(", "Do not create a dataframe inside the code."),
    ]
    for pattern, msg in checks:
        if re.search(pattern, code):
            return msg
    return None


@tool
def python_repl(
        code: Annotated[str, "The Python code to execute"],
        dataset_key: Annotated[str, "Registry key from file loader"] = "",
) -> str:
    """
    This is a Python tool that executes Python code on loaded
    tabular dataframes.

    Runtime contract:
    - If `dataset_key` is provided, the tool loads the dataframe from the dataset registry.
    - The dataframe is exposed to the code as `df`.
    - Generated code must use `df` directly.
    - Do not create or reassign `df`.
    - Assign the final answer to a variable named `result`.
    """

    if dataset_key:
        df = DATASET_REGISTRY.get(dataset_key)
        if df is None:
            return f"Dataset '{dataset_key}' not found"

    if len(code) > 1000:
        # if `code` contains forceful data reads, it will become too long
        return (
            "Code generation error: code is too long. "
            "Do not inline dataset values. Use `df` directly."
        )

    violation = _violates_code_rules(code)
    if violation:
        return f"Code generation error: {violation}"

    local_vars = {"df": df}
    safe_globals = {"pd": pd}

    try:
        compile(code, "<string>", "exec")
        exec(code, safe_globals, local_vars)
        return str(local_vars.get("result", "Exectution complete"))
    except Exception as e:
        return f"Execution error: {e}"