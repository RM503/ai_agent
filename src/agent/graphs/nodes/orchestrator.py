from __future__ import annotations

from agent.graphs.router import decide_route
from agent.schemas.graph_state import AgentState

def orchestrator_node(state: AgentState) -> dict:
    file_name = None
    if state.uploaded_artifact is not None:
        file_name = state.uploaded_artifact.file_name

    route = decide_route(state.user_message, file_name)
    return {"route": route}