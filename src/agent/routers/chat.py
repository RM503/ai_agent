# Route for chat UI
from fastapi import APIRouter

from agent.graphs.builder import build_graph
from agent.schemas.api import ChatRequest, ChatResponse
from agent.schemas.graph_state import AgentState

# Define API route & build graph
chat_router = APIRouter()
graph = build_graph()

@chat_router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """
    This router handles inputs coming from the frontend's
    chat UI, creates state for LangGraph and passes it to
    the compiled graph for response.

    Args:
        request (ChatRequest): request object from frontend
    Returns:
        ChatResponse: response object
    """
    state = AgentState(
        session_id=request.session_id,
        user_message=request.message
    )

    result = graph.invoke(state)

    return ChatResponse(
        session_id=request.session_id,
        route=result.get("route", "general"),
        response=result.get("response_text")
    )