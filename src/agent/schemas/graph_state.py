from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal, Optional
from uuid import UUID

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
    summary: Optional[str] = None
    preview_rows: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    chart_paths: list[str | Path] = Field(default_factory=list)

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

    # Agent response
    response_text: Optional[str] = None

    # Conversation memory
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)

    # Dataset key for inline data
    dataset_key: Optional[str | UUID] = None
    metadata: dict[str, Any] = Field(default_factory=dict)