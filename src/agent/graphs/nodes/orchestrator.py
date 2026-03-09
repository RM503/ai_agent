from __future__ import annotations

from langchain_core.messages import BaseMessage, HumanMessage

from agent.graphs.router import decide_route
from agent.schemas.graph_state import AgentState

def get_latest_user_message(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content
    return ""

def orchestrator_node(state: AgentState) -> dict:
    """
    You are the 'Orchestrator node'. You decide which node to
    direct the state to depending on what the user wants using
    the `decide_route` function.
    """
    file_name = None
    if state.uploaded_artifact is not None:
        file_name = state.uploaded_artifact.file_name

    route = decide_route(
        get_latest_user_message(state.messages),
        file_name
    )
    return {"route": route}