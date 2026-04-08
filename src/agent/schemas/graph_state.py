from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Optional
from uuid import UUID

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

class UploadedArtifact(BaseModel):
    """Class for uploads"""
    file_id: Optional[str] = None
    file_name: Optional[str] = None
    file_content: Optional[str] = None
    file_path: Optional[str | Path] = None

class AnalysisResult(BaseModel):
    dataset_key: Optional[str] = None
    status: Optional[str] = None
    result_type: Optional[str] = None
    result_value: Optional[Any] = None
    generated_at: Optional[datetime] = None
    summary: Optional[str] = None
    preview_rows: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    chart_paths: list[str | Path] = Field(default_factory=list)

class RetrievedDocument(BaseModel):
    """Class for retrieved documents"""
    content: str
    source: Optional[str] = None # e.g. file name, URL etc
    doc_id: Optional[str | UUID] = None # e.g. vector store document id
    chunk_idx : Optional[int] = None
    score : Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

class AgentState(BaseModel):
    """The general agent state"""

    # Session
    session_id: str | UUID

    # Routing decision
    route: Optional[
        Literal["transcription", "summarization", "data_analysis", "general"]
    ] = None

    # Uploaded artifact context
    # allows one uploaded artifact at time
    uploaded_artifacts: Optional[UploadedArtifact] = None

    # Working artifacts
    transcript_text: Optional[str] = None
    summary_text: Optional[str] = None
    analysis_result: Optional[AnalysisResult] = None

    # Conversation memory
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)

    # Dataset key for inline data
    dataset_key: Optional[str | UUID] = None
    metadata: dict[str, Any] = Field(default_factory=dict)