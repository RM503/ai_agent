from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import tools_condition

from agent.graphs.nodes.data_analysis import data_analysis_node
from agent.graphs.nodes.orchestrator import orchestrator_node
from agent.graphs.nodes.responder import general_responder_node
from agent.graphs.nodes.tool_executor import tool_executor_node
from agent.schemas.graph_state import AgentState

def _route_after_orchestrator(state: AgentState) -> str:
    """Return route decided by orchestrator."""
    return state.route or "general"

def _route_back_from_tools(state: AgentState) -> str:
    """
    Returns to the route after tool execution.
    """
    return state.route or "general"

def build_graph() -> CompiledStateGraph:
    """Build a compiled graph from usable nodes"""
    # Create checkpoint for memory
    checkpointer = InMemorySaver()
    # tools = [get_weather, web_search]

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("general", general_responder_node)
    graph.add_node("data_analysis", data_analysis_node)
    graph.add_node("tools", tool_executor_node)

    # Add edges and conditional edges

    # Start
    graph.add_edge(START, "orchestrator")

    # Orchestrator routing
    graph.add_conditional_edges(
        "orchestrator",
        _route_after_orchestrator,
        {
            "general": "general",
            "data_analysis": "data_analysis"
        }
    )

    # General tool calling or END
    graph.add_conditional_edges(
        "general",
        tools_condition,
        {
            "tools": "tools",
            END: END
        }
    )

    # Data analysis tool calling or END
    graph.add_conditional_edges(
        "data_analysis",
        tools_condition,
        {
            "tools": "tools",
            END: END
        }
    )

    # After tool execution, return to the selected route
    graph.add_conditional_edges(
        "tools",
        _route_back_from_tools,
        {
            "general": "general",
            "data_analysis": "data_analysis"
        }
    )

    return graph.compile(checkpointer=checkpointer)