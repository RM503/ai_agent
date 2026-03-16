import textwrap

# def build_wrapped_code(code: str, dataset_path: str) -> str:
#     code = textwrap.dedent(code).strip()
#
#     wrapped_code = textwrap.dedent(f"""
#                 import base64
#                 import io
#                 import json
#                 import numpy as np
#                 import pandas as pd
#                 import matplotlib
#                 matplotlib.use("Agg") # non-interactive plotting backend
#                 import matplotlib.pyplot as plt
#
#                 # Suppress plt.show() in use code - prevent figure clearing
#                 plt.show = lambda *a, **kw: None
#
#                 df = pd.read_csv("{dataset_path}")
#
#                 # Use code starts here
#                 {code}
#                 # User code ends here
#
#                 print("DEBUG: user code finished")
#
#                 # Capture all open figures before any result serialization
#                 _charts = []
#                 for _fig_num in plt.get_fignums():
#                     _fig = plt.figure(_fig_num)
#                     _buf = io.BytesIO()
#                     _fig.savefig(_buf, format="png", bbox_inches="tight", dpi=120)
#                     _buf.seek(0)
#                     _charts.append(base64.b64encode(_buf.read()).decode("utf-8"))
#                     plt.close(_fig)
#
#                 if _charts:
#                     print("__CHARTS__" + json.dumps(_charts))
#
#                 print("DEBUG: locals keys =", list(locals().keys()))
#
#                 if "result" not in locals():
#                     if plt.get_fignums():
#                         result = "plot generated"
#                     else:
#                         raise ValueError("Your code must assign the final answer to a variable named `result`.")
#
#                 print("DEBUG: result type =", type(result).__name__)
#
#                 def _serialize_result(obj):
#                     # Serialize results into JSON for LangGraph
#                     if isinstance(obj, pd.DataFrame):
#                         return {{
#                             "type": "dataframe",
#                             "value": obj.head(100).to_dict(orient="records"),
#                             "columns": obj.columns.tolist(),
#                             "shape": list(obj.shape),
#                         }}
#                     if isinstance(obj, pd.Series):
#                         return {{
#                             "type": "series",
#                             "value": obj.head(100).to_dict(),
#                             "length": int(obj.shape[0]),
#                         }}
#                     if isinstance(obj, np.ndarray):
#                         return {{
#                             "type": "ndarray",
#                             "value": obj.tolist(),
#                             "shape": list(obj.shape),
#                         }}
#                     try:
#                         json.dumps(obj)
#                         return {{
#                             "type": "scalar",
#                             "value": obj,
#                         }}
#                     except TypeError:
#                         return {{
#                             "type": type(obj).__name__,
#                             "value": str(obj),
#                         }}
#
#                 payload = _serialize_result(result)
#                 print("DEBUG: payload ready")
#                 print("__RESULT__" + json.dumps(payload))
#                 print("DEBUG: wrapper finished")
#                 """).strip()
#     return wrapped_code

def build_wrapped_code(code: str, dataset_path: str) -> str:
    code_lines = textwrap.dedent(code).strip().splitlines()

    header = [
        "import json",
        "import io as _io",
        "import base64 as _base64",
        "import pandas as pd",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "",
        "plt.show = lambda *a, **kw: None",
        "",
        f'df = pd.read_csv("{dataset_path}")',
        "",
        "# User code starts here",
    ]

    footer = [
        "# User code ends here",
        "",
        "_charts_b64 = []",
        "for _fig_num in plt.get_fignums():",
        "    _fig = plt.figure(_fig_num)",
        "    _buf = _io.BytesIO()",
        '    _fig.savefig(_buf, format="png", bbox_inches="tight", dpi=120)',
        "    _buf.seek(0)",
        "    _charts_b64.append(_base64.b64encode(_buf.read()).decode('utf-8'))",
        "    plt.close(_fig)",
        "",
        "if _charts_b64:",
        '    print("__CHARTS__" + json.dumps(_charts_b64), flush=True)',
        "",
        'print("DEBUG: fignums =", plt.get_fignums(), flush=True)',
        'print("DEBUG: locals =", list(locals().keys()), flush=True)',
        "",
        'if "result" not in locals():',
        "    if _charts_b64:",
        '        result = "plot generated"',
        "    else:",
        '        raise ValueError("Your code must assign the final answer to a variable named `result`.")',
        "",
        'print("DEBUG: result type =", type(result).__name__, flush=True)',
        "",
        "def _serialize_result(obj):",
        "    if isinstance(obj, pd.DataFrame):",
        '        return {"type": "dataframe", "value": obj.head(100).to_dict(orient="records"), "columns": obj.columns.tolist(), "shape": list(obj.shape)}',
        "    if isinstance(obj, pd.Series):",
        '        return {"type": "series", "value": obj.head(100).to_dict(), "length": int(obj.shape[0])}',
        "    if isinstance(obj, np.ndarray):",
        '        return {"type": "ndarray", "value": obj.tolist(), "shape": list(obj.shape)}',
        "    try:",
        "        json.dumps(obj)",
        '        return {"type": "scalar", "value": obj}',
        "    except TypeError:",
        '        return {"type": type(obj).__name__, "value": str(obj)}',
        "",
        "payload = _serialize_result(result)",
        'print("__RESULT__" + json.dumps(payload), flush=True)',
        'print("DEBUG: wrapper finished", flush=True)',
    ]

    all_lines = header + code_lines + footer
    return "\n".join(all_lines)