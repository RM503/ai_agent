# Python REPL tool
import json
import re
import os
import tempfile
from typing import Annotated

from e2b_code_interpreter import Sandbox
from langchain.tools import tool

from .dataset_registry import DATASET_REGISTRY
from .repl_wrapper import build_wrapped_code
from agent.common.logging_config import get_logger

logger = get_logger(__name__)

def _violates_code_rules(code: str) -> str | None:
    """
    This function checks if the given code violates any code rules.
    """
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
    A Python execution tool that runs code against loaded tabular dataframes.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    DATAFRAME CONTRACT
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    - If `dataset_key` is provided, the dataframe is pre-loaded and available as `df`.
    - Use `df` directly for all analysis. Never reload, recreate, or reassign it.
    - Never read files from disk (no pd.read_csv, pd.read_excel, etc.).
    - Never inline dataset rows or values into the code.
    - Never parse JSON or literals (no json.loads, ast.literal_eval).
    - Never create a new DataFrame with pd.DataFrame(...).

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    RESULT CONTRACT
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    - You MUST assign the final answer to a variable named `result`.
    - Do not rely on implicit output or bare expressions.
    - Do not only print the answer.

    Correct examples:
        result = df.head()
        result = df.describe()
        result = df[df["column"] > 700]
        result = df["column"].mean()

    Incorrect examples:
        df.head()                            # bare expression, no assignment
        filtered = df[df["column"] > 700]   # assigned but not to `result`
        print(df.describe())                 # print only, no assignment

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    PLOTTING
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    - You may use matplotlib to produce charts. Import and use it normally.
    - Do NOT call plt.show(), plt.savefig(), or plt.close() — the runtime handles all of these.
    - For pure-plot tasks with no tabular result, assign: result = "plot generated"
    - You may produce multiple figures; all open figures are captured automatically.

    Correct plotting example:
        import matplotlib.pyplot as plt

        plt.hist(df["annual_income"], bins=20, color="steelblue", edgecolor="black")
        plt.title("Distribution of Annual Income")
        plt.xlabel("Annual Income")
        plt.ylabel("Frequency")

        result = "plot generated"

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    OUTPUT TYPES
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    - For filtered tables: assign the resulting DataFrame to `result`
    - For aggregations/summaries: assign the scalar, Series, or summary object to `result`
    - For plots: assign result = "plot generated"
    - For multi-step analysis with a plot: assign the primary computed value to `result`
    """
    api_key = os.getenv("E2B_SANDBOX_API_KEY")
    if not api_key:
        return json.dumps({
            "status": "error",
            "error": "Missing E2B_SANDBOX_API_KEY."
        })

    if not code or not code.strip():
        return json.dumps({
            "status": "error",
            "error": "No code provided."
        })

    if len(code) > 1000:
        # if `code` contains forceful data reads, it will become too long
        return json.dumps({
            "status": "error",
            "error": (
                "Generated code is too long. "
                "Do not inline dataset values. Use `df` directly."
            )
        })

    violation = _violates_code_rules(code)
    if violation:
        return json.dumps({
            "status": "error",
            "error": f"Code generation error: {violation}"
        })

    df = DATASET_REGISTRY.get(dataset_key)
    if df is None:
        return json.dumps({
            "status": "error",
            "error": f"Dataset '{dataset_key}' not found."
        })

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name
            df.to_csv(tmp_path, index=False)

        with Sandbox.create(api_key=api_key) as sandbox:
            remote_path = "/tmp/input.csv"
            with open(tmp_path, "rb") as f:
                sandbox.files.write(remote_path, f.read())

            wrapped_code = build_wrapped_code(code, remote_path) # generates f-string for code to go inside the sandbox
            execution = sandbox.run_code(wrapped_code)

            if execution.error:
                return json.dumps({
                    "status": "error",
                    "error": f"{execution.error.name}: {execution.error.value}\n{execution.error.traceback}",
                })

            # Retrieve standard output and error from logs
            stdout_lines = getattr(execution.logs, "stdout", []) or []
            stderr_lines = getattr(execution.logs, "stderr", []) or []

            stdout_text = "\n".join(stdout_lines)
            stderr_text = "\n".join(stderr_lines)

            result_payload = None
            charts = []
            for line in stdout_text.splitlines():
                if line.startswith("__RESULT__"):
                    result_payload = json.loads(line[len("__RESULT__"):])
            charts_marker = "__CHARTS__"
            charts_idx = stdout_text.find(charts_marker)
            if charts_idx != -1:
                # Find end of the JSON array (next newline or end of string)
                json_start = charts_idx + len(charts_marker)
                json_end = stdout_text.find("\n", json_start)
                raw_json = stdout_text[json_start:] if json_end == -1 else stdout_text[json_start:json_end]
                try:
                    raw = json.loads(raw_json.strip())
                    charts = [{"mime": "image/png", "data": b64} for b64 in raw]
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse __CHARTS__ payload: {e}")

            status = "ok" if not stderr_text else "partial"

        return json.dumps({
            "status": status,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "result": result_payload,
            "charts": charts
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
        })
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass