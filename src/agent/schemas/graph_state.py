"""
Module containing the various graph states to be used by the LangGraph
agent, including the various nodes, tools and the AgentState.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import UUID

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

class UploadedArtifact(BaseModel):
    """Class for uploads"""
    file_id: str | None = None
    file_name: str | None = None
    file_content: str | None = None
    file_path: str | Path | None = None

class AnalysisResult(BaseModel): 
    dataset_key: str | None = None
    status: str | None = None
    result_type: str | None = None
    result_value: str | None = None
    generated_at: datetime | None = None
    summary: str | None = None
    preview_rows: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    chart_paths: list[str | Path] = Field(default_factory=list)

# === RAG === #
class RagScope(BaseModel):
    """Defines scope variables for RAG operation"""
    scope: Literal["global", "user", "project", "session"] = "global"
    user_id : str | UUID | None = None
    project_id: str | UUID | None = None
    collection: str | None = None
    ingestion_id: str | UUID | None = None

class RagIngestionRef(BaseModel):
    """Reference to async RAG ingestion job available to this session."""
    ingestion_id: str | UUID
    session_id: str | UUID
    user_id: str | UUID | None = None
    status: Literal["queued", "running", "done", "error"] | None = None
    file_count: int | None = None
    chunk_count: int | None = None
    document_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

class RagQuery(BaseModel):
    """Retrieval query derived from user input, used for RAG retrieval."""
    original_query: str
    rewritten_query: str | None = None
    needs_retrieval: bool = False
    query_type: Literal["qa", "summary", "comparison", "lookup", "other"] = "qa"
    filters: dict[str, Any] = Field(default_factory=dict)

class RetrievedDocument(BaseModel):
    """Retrieved vector-store chunk"""
    content: str
    source: str | None = None
    doc_id: str | UUID | None = None
    ingestion_id: str | UUID | None = None
    file_name: str | None = None
    page: int | None = None
    chunk_idx: int | None = None
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

class RagContext(BaseModel):
    """Final evidence context passed to the answering node."""
    query: RagQuery | None = None
    documents: list[RetrievedDocument] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_status: Literal["not_run", "found", "empty", "error"] = "not_run"
    error: str | None = None

class AgentState(BaseModel):
    """The general agent state"""

    # Session
    session_id: str | UUID

    # Routing decision
    route: Literal[
        "transcription",
        "summarization",
        "data_analysis",
        "rag",
        "general"
    ] | None = None

    # Uploaded artifact context
    # allows one uploaded artifact at time
    uploaded_artifacts: UploadedArtifact | None = None

    # RAG state
    rag_scope: RagScope = Field(default_factory=RagScope)
    rag_ingestions: list[RagIngestionRef] = Field(default_factory=list)
    active_ingestion_id: str | UUID | None = None
    rag_context: RagContext | None = None

    # Working artifacts
    transcript_text: str | None = None
    summary_text: str | None = None
    analysis_result: AnalysisResult | None = None

    # Conversation memory
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)

    # Dataset key for inline data
    dataset_key: str | UUID | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
