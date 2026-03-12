from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage

from agent.schemas.graph_state import AgentState
from agent.tools.data_analysis.file_loader import file_loader
from agent.tools.data_analysis.pythonrepl import python_repl
from agent.tools.weather import get_weather
from agent.tools.web_search import web_search

tools_list = [
    file_loader,
    get_weather,
    python_repl,
    web_search,
]

tool_map = {t.name: t for t in tools_list}

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
    for tool_call in last_message.tool_calls:
        tool = tool_map[tool_call["name"]]
        try:
            output = tool.invoke(tool_call["args"])
        except Exception as e:
            output = f"Error: {e}"

        results.append(
            ToolMessage(
                content=str(output),
                tool_call_id=tool_call["id"]
            )
        )
    return {"messages": results}