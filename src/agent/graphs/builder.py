from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from agent.graphs.nodes.orchestrator import orchestrator_node
from agent.graphs.nodes.responder import general_responder_node
from agent.schemas.graph_state import AgentState

def _route_after_orchestrator(state: AgentState) -> str:
    return state.route or "general"

def build_graph() -> CompiledStateGraph:
    """Build a compiled graph from usable nodes"""
    # Create checkpoint for memory
    checkpointer = InMemorySaver()

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("general", general_responder_node)

    graph.add_edge(START, "orchestrator")
    graph.add_edge("orchestrator", "general")

    graph.add_edge("general", END)

    return graph.compile(checkpointer=checkpointer)