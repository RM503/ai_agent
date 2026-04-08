"""
FASTAPI route for LLM chat
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator
from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from sqlmodel import Session

from .router_utils import retrieve_uploaded_artifacts
from agent.common.db import get_session
from agent.common.logging_config import get_logger
from agent.graphs.builder import build_graph
from agent.repositories.chat_repository import ChatRepository
from agent.schemas.api import ChatRequest
from agent.schemas.graph_state import AgentState, AnalysisResult, UploadedArtifact

logger = get_logger(__name__)

# Define API route & build graph
router: APIRouter = APIRouter()
graph: CompiledStateGraph = build_graph()


def _coerce_uploaded_artifact(payload: dict[str, Any] | str | None) -> UploadedArtifact | None:
    """Convert redis payload into UploadedArtifact when present."""
    if not payload:
        return None
    if isinstance(payload, str):
        return UploadedArtifact.model_validate_json(payload)
    return UploadedArtifact.model_validate(payload)


def _coerce_analysis_result(payload: dict[str, Any] | AnalysisResult | None) -> AnalysisResult | None:
    """Convert checkpoint payload into AnalysisResult when present."""
    if not payload:
        return None
    if isinstance(payload, AnalysisResult):
        return payload
    return AnalysisResult.model_validate(payload)

@router.post("/chat", response_class=StreamingResponse)
async def chat(request: ChatRequest, db: Session=Depends(get_session)) -> StreamingResponse:
    """
    This route is used to generate streamed chat messages for the chat UI.

    Args:
        request: ChatRequest object containing the session ID and message

    Returns:
        StreamingResponse object containing the streamed chat messages
    """
    session_id: str = str(request.session_id)
    config: dict[str, Any] = {"configurable": {"thread_id": session_id}}

    # DB insert for human messages
    repo = ChatRepository(
        engine=db.get_bind(),
        session=db,
        session_id=UUID(session_id),
        user_id=None
    )

    repo.ensure_chat_session()
    repo.insert_chat_message(role="human", content=request.message, metadata={})

    logger.info(f"session_id:{session_id}")

    # Check for file uploads
    uploaded_artifact_payload: dict[str, Any] | str | None = await retrieve_uploaded_artifacts(session_id)
    uploaded_artifact = _coerce_uploaded_artifact(uploaded_artifact_payload)

    snapshot = graph.get_state(config)
    prior_values: dict[str, Any] = snapshot.values if snapshot else {}

    state: AgentState = AgentState(
        session_id=session_id,
        messages=[HumanMessage(content=request.message)],
        route=prior_values.get("route"),
        uploaded_artifacts=uploaded_artifact or _coerce_uploaded_artifact(prior_values.get("uploaded_artifacts")),
        transcript_text=prior_values.get("transcript_text"),
        summary_text=prior_values.get("summary_text"),
        analysis_result=_coerce_analysis_result(prior_values.get("analysis_result")),
        dataset_key=prior_values.get("dataset_key"),
        metadata=prior_values.get("metadata", {}),
    )


    # Define invocation payload - important to add all the necessary information for the agent to work
    invoke_payload: dict[str, Any] = {
        "session_id": session_id,
        "messages": [HumanMessage(content=request.message)],
        "uploaded_artifacts": state.uploaded_artifacts,
        "route": state.route,
        "transcript_text": state.transcript_text,
        "summary_text": state.summary_text,
        "analysis_result": state.analysis_result,
        "dataset_key": state.dataset_key,
        "metadata": state.metadata
    }

    async def event_generator() -> AsyncIterator:
        """Asynchronous event generator for streaming chat messages"""
        payload_start: str = json.dumps({"type": "start", "session_id": session_id}) + "\n"
        yield payload_start

        # Store chat chunks and artifacts
        chunks: list[str] = []
        charts: list[str] = []

        async for event in graph.astream_events(
            invoke_payload,
            config=config,
            version="v2"
        ):
            # event will be a JSON response with an 'event' key
            # search for 'on_chat_model_stream' for chat response
            kind: str = event["event"]
            if kind == "on_chat_model_stream":
                chunk: str = event["data"]["chunk"].content      # AI message from stream
                node : str = event["metadata"]["langgraph_node"] # LangGraph node that is working
                if chunk and node != "orchestrator":
                    # Define payload with the chunk and the node that is working
                    payload_main: str = json.dumps({
                        "type": "token",
                        "content": chunk,
                        "session_id": session_id,
                        "node": node
                    })+ "\n"
                    chunks.append(chunk)
                    yield f"{payload_main}"

            # search for 'on_tool_end' for generated plots
            elif kind == "on_tool_end":
                try:
                    output: str = event["data"].get("output", "")
                    tool_result: Any = json.loads(output)
                    tool_charts: list[str] = tool_result.get("charts", [])

                    if tool_charts:
                        charts.extend(tool_charts)
                except (json.JSONDecodeError, AttributeError, TypeError) as e:
                    logger.warning(f"on_tool_end parse failed: {e}")
                    pass

        if charts:
            logger.info(f"Yielding {len(charts)} chart(s) to SSE stream")
            yield json.dumps({"type": "charts", "charts": charts, "session_id": session_id}) + "\n"
        else:
            logger.warning("No charts to yield at end of event_generator")

        final_snapshot = graph.get_state(config)
        final_values: dict[str, Any] = final_snapshot.values if final_snapshot else {}
        final_analysis_result = final_values.get("analysis_result")
        if isinstance(final_analysis_result, AnalysisResult):
            final_analysis_result = final_analysis_result.model_dump(mode="json")

        logger.info(f"{json.dumps(state.model_dump(mode="json"), indent=4)})")

        full_text  : str = "".join(chunks)

        # DB insert for ai messages
        repo.insert_chat_message(role="ai", content=full_text, metadata={})

        payload_end: str = json.dumps({
            "type": "end",
            "session_id": session_id,
            "response": full_text,
            "dataset_key": final_values.get("dataset_key"),
            "analysis_result": final_analysis_result,
        }) + "\n"

        yield f"{payload_end}"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
