from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from json.decoder import JSONDecodeError
from typing import Any, Callable, Optional

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.tool import ToolCall

from agent.schemas.graph_state import AgentState, AnalysisResult
from agent.tools.data_analysis.file_loader import file_loader
from agent.tools.data_analysis.python_repl import python_repl
from agent.tools.weather import get_weather
from agent.tools.web_search import web_search

tools_list = [
    file_loader,
    get_weather,
    python_repl,
    web_search,
]

tool_map = {t.name: t for t in tools_list}

@dataclass(frozen=True, slots=True)
class ToolOutputContext:
    """
    Implements output context for tool call

    Attributes:
        state (AgentState): the graph's state
        tool_call (ToolCall): the tool that was called in previous message
        output_dict (dict[str, Any]): output produced by the tool invocation removed of chart blobs (used as LLM output)
        raw_output_dict (dict[str, Any]): raw snapshot of the output with complete information
    """
    state: AgentState
    tool_call: ToolCall
    output_dict: dict[str, Any]
    raw_output_dict: dict[str, Any]

@dataclass
class ToolHandlerUpdates:
    dataset_key: Optional[str] = None
    analysis_result: Optional[AnalysisResult] = None

ToolOutputHandler = Callable[[ToolOutputContext], ToolHandlerUpdates]

def _after_file_loader(ctx: ToolOutputContext) -> ToolHandlerUpdates:
    """State update after `file_loader` use"""
    return ToolHandlerUpdates(dataset_key=ctx.output_dict.get("dataset_key"))

def _after_python_repl(ctx: ToolOutputContext) -> ToolHandlerUpdates:
    """State update after `python_repl` use"""
    output_dict = ctx.output_dict
    raw_output_dict = ctx.raw_output_dict
    result_payload = output_dict.get("result") or {}
    raw_charts = raw_output_dict.get("charts", [])
    result_type = result_payload.get("type")
    result_value = result_payload.get("value")
    preview_rows: list[dict[str, object]] = []
    metrics: dict[str, object] = {}

    if result_type == "dataframe" and isinstance(result_value, list):
        preview_rows = result_value
        if result_payload.get("shape") is not None:
            metrics["shape"] = result_payload["shape"]
        if result_payload.get("columns") is not None:
            metrics["columns"] = result_payload["columns"]
    elif result_type == "series" and result_payload.get("length") is not None:
        metrics["length"] = result_payload["length"]

    analysis_result = AnalysisResult(
        dataset_key=ctx.tool_call["args"].get("dataset_key") or ctx.state.dataset_key,
        status=output_dict.get("status"),
        result_type=result_type,
        result_value=result_value,
        generated_at=datetime.now(UTC).isoformat(),
        preview_rows=preview_rows,
        metrics=metrics,
        chart_paths=[
            f"inline_chart_{idx}"
            for idx, _chart in enumerate(raw_charts, start=1)
        ],
    )

    return ToolHandlerUpdates(analysis_result=analysis_result)


TOOL_OUTPUT_HANDLERS: dict[str, ToolOutputHandler] = {
    "file_loader": _after_file_loader,
    "python_repl": _after_python_repl,
}

def tool_executor_node(state: AgentState) -> dict:
    """
    This function runs the tool executor. This is generally used
    by other nodes to decide which tool to use based on the previous
    message.
    """
    # Check previous message for tool calls
    last_message = state.messages[-1]

    if not isinstance(last_message, AIMessage):
        return {"messages": []}

    results = []
    dataset_key: str | None = None
    analysis_result: AnalysisResult | None = None

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_id = tool_call["id"]
        tool = tool_map[tool_name]

        try:
            # Manually invoke tool from tool call arguments
            output = tool.invoke(tool_call["args"])
            
        except Exception as e:
            output = f"Error: {e}"

        output_dict = None
        raw_output_dict = None

        # Strip chart blobs before they enter the LLM context window
        try:
            output_dict = json.loads(output)
            raw_output_dict = dict(output_dict)
            has_charts = bool(output_dict.pop("charts", []))
            output_dict["has_charts"] = has_charts
            llm_output = json.dumps(output_dict)
        except (JSONDecodeError, AttributeError, TypeError):
            llm_output = str(output)

        handler = TOOL_OUTPUT_HANDLERS.get(tool_name)
        if (
            handler is not None
            and isinstance(output_dict, dict)
            and isinstance(raw_output_dict, dict)
        ):
            updates = handler(
                ToolOutputContext(
                    state=state,
                    tool_call=tool_call,
                    output_dict=output_dict,
                    raw_output_dict=raw_output_dict,
                )
            )
            if updates.dataset_key is not None:
                dataset_key = updates.dataset_key or dataset_key
            if updates.analysis_result is not None:
                analysis_result = updates.analysis_result

        results.append(ToolMessage(content=llm_output, tool_call_id=tool_id))

    update = {"messages": results}
    if dataset_key is not None:
        update["dataset_key"] = dataset_key
    if analysis_result is not None:
        update["analysis_result"] = analysis_result
    return update
