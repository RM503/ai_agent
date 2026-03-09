# Route for chat UI
from fastapi import APIRouter

from langchain_core.messages import HumanMessage

from agent.graphs.builder import build_graph
from agent.schemas.api import ChatRequest, ChatResponse
from agent.schemas.graph_state import AgentState

# Define API route & build graph
router = APIRouter()
graph = build_graph()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    This router handles inputs coming from the frontend's
    chat UI, creates state for LangGraph and passes it to
    the compiled graph for response.

    Args:
        request (ChatRequest): request object from frontend
    Returns:
        ChatResponse: response object
    """

    state = AgentState(session_id=request.session_id)

    result = graph.invoke(
    {
            "session_id": state.session_id,
            "messages": [HumanMessage(content=request.message)],
        },
        config={"configurable": {"thread_id": str(state.session_id)}},
    )

    response = result.get("response_text")
    if not response:
        response = result["messages"][-1]

    return ChatResponse(
        session_id=request.session_id,
        route=result.get("route", "general"),
        response=response
    )