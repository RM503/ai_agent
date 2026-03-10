# Route for chat UI
import json
from typing import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from langchain_core.messages import HumanMessage

from agent.common.logging_config import get_logger
from agent.graphs.builder import build_graph
from agent.schemas.api import ChatRequest
from agent.schemas.graph_state import AgentState

logger = get_logger(__name__)

# Define API route & build graph
router = APIRouter()
graph = build_graph()

@router.post("/chat", response_class=StreamingResponse)
async def chat(request: ChatRequest) -> StreamingResponse:
    """
    Route for generating streamed chat messages.
    """
    session_id = request.session_id
    state = AgentState(session_id=session_id)

    async def event_generator() -> AsyncIterator:
        payload_start = json.dumps({"type": "start", "session_id": session_id})
        #logger.info(f"{payload_start}")
        yield payload_start

        chunks = []
        async for event in graph.astream_events(
                {
                    "session_id": session_id,
                    "messages": [HumanMessage(content=request.message)]
                },
            config={"configurable": {"thread_id": str(state.session_id)}},
            version="v2"
        ):
            # event will be a JSON response with an 'event' key
            # search for 'on_chat_model_stream'
            kind = event["event"]
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"].content # AI message from stream
                node  = event["metadata"]["langgraph_node"] # LangGraph node that is working
                if chunk:
                    # Define payload with
                    payload_main = json.dumps({
                        "type": "token",
                        "content": chunk,
                        "session_id": session_id,
                        "node": node
                    })
                    chunks.append(chunk)
                    yield f"{payload_main}"

        full_text = "".join(chunks)
        payload_end = json.dumps({"type": "end", "session_id": session_id, "response": full_text})

        yield f"{payload_end}"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )