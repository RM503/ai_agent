from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import UUID

from langchain_core.documents import Document
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

class RetrievedDocument(BaseModel):
    """Class for retrieved documents"""
    content: str
    source: str | None = None # e.g. file name, URL etc
    doc_id: str | UUID | None = None # e.g. vector store document id
    chunk_idx : int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

class AgentState(BaseModel):
    """The general agent state"""

    # Session
    session_id: str | UUID

    # Routing decision
    route: Literal["transcription", "summarization", "data_analysis", "general"] | None = None

    # Uploaded artifact context
    # allows one uploaded artifact at time
    uploaded_artifacts: UploadedArtifact | None = None

    # Working artifacts
    transcript_text: str | None = None
    summary_text: str | None = None
    analysis_result: AnalysisResult | None = None

    # Conversation memory
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)

    # Dataset key for inline data
    dataset_key: str | UUID | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
