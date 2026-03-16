# Route for chat UI
import json
from typing import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from langchain_core.messages import HumanMessage

from .router_utils import retrieve_uploaded_artifacts
from agent.common.logging_config import get_logger
from agent.graphs.builder import build_graph
from agent.schemas.api import ChatRequest
from agent.schemas.graph_state import AgentState, UploadedArtifact
from agent.tools.data_analysis.file_loader import file_loader

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

    logger.info(f"session_id:{session_id}")

    # Check for file uploads
    uploaded_artifact = await retrieve_uploaded_artifacts(session_id)

    if uploaded_artifact:
        state.uploaded_artifacts = UploadedArtifact.model_validate_json(uploaded_artifact)

    if state.uploaded_artifacts and state.uploaded_artifacts.file_path:
        file_loader.invoke({
            "file_path": str(state.uploaded_artifacts.file_path)
        })

    logger.info(f"{json.dumps(state.model_dump(), indent=4)})")

    # Define invocation payload
    invoke_payload = {
        "session_id": session_id,
        "messages": [HumanMessage(content=request.message)],
        "uploaded_artifacts": state.uploaded_artifacts,
        "route": state.route,
        "transcript_text": state.transcript_text,
        "summary_text": state.summary_text,
        "analysis_result": state.analysis_result,
        "response_text": state.response_text,
        "dataset_key": state.dataset_key,
        "metadata": state.metadata
    }

    async def event_generator() -> AsyncIterator:
        payload_start = json.dumps({"type": "start", "session_id": session_id})
        #logger.info(f"{payload_start}")
        yield payload_start

        chunks = []
        charts = []
        async for event in graph.astream_events(
            invoke_payload,
            config={"configurable": {"thread_id": str(state.session_id)}},
            version="v2"
        ):
            # event will be a JSON response with an 'event' key
            # search for 'on_chat_model_stream' for chat response
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

            # search for 'on_tool_end' for generated plots
            elif kind == "on_tool_end":
                try:
                    output = event["data"].get("output", "")
                    tool_result = json.loads(output)
                    tool_charts = tool_result.get("charts", [])

                    if tool_charts:
                        charts.extend(tool_charts)
                except (json.JSONDecodeError, AttributeError, TypeError):
                    pass

        if charts:
            yield json.dumps({"type": "charts", "charts": charts, "session_id": session_id})

        full_text = "".join(chunks)
        payload_end = json.dumps({"type": "end", "session_id": session_id, "response": full_text})

        yield f"{payload_end}"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )