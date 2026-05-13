"""
This module implements a routing function that is used by the
orchestrator node.
"""

from __future__ import annotations

from pathlib import Path

from langgraph.graph import END

from agent.schemas.graph_state import AgentState

# Audio extensions for transcription tasks
AUDIO_EXTS: set[str] = {".mp3", ".wav", ".m4a", ".mp4", ".aac", ".flac"}
# Data file extensions for analysis
DATA_EXTS: set[str] = {".csv", ".xlsx", ".xls", ".parquet"}
# Document extensions for RAG
RAG_EXTS: str[str] = {".pdf", ".txt", ".docx"}

# Possible tasks list
ANALYSIS_TASKS: list[str] = [
    "analyze dataset", "analyze csv", "data analysis", "summarize dataframe", "profile this data"
]
SUMMARY_TASKS: list[str] = ["summarize", "meeting notes", "minutes", "summary"]
TRANSCRIPTION_TASKS: list[str] = ["transcribe", "transcription", "audio", "meeting recording"]
RAG_TASKS: list[str] = [
    "document",
    "uploaded document",
    "uploaded file",
    "pdf",
    "notes",
    "knowledge base",
    "based on the file",
    "based on the document",
    "in the document",
    "according to the document",
    "what does the document say"
]

def decide_route(
        message: str,
        file_name: str | None=None,
        has_rag_ingestions: bool=False
    ) -> str:
    """
    This function implements a router for the orchestrator for delegating
    tasks to sub-agents from message contexts. As of now, the context might be
    narrow.

    Args:
        message (str): the message to be processed.
        file_name (str, optional): name of file (if passed)
        has_rag_ingestions (bool): if active data ingestion is present
    Returns:
        str: the route identifier.
    """
    message = message.lower()
    suffix = Path(file_name).suffix.lower() if file_name else ""

    if suffix in AUDIO_EXTS or any(k in message for k in TRANSCRIPTION_TASKS):
        return "transcription"
    if  suffix in DATA_EXTS or any(k in message for k in ANALYSIS_TASKS):
        return "data_analysis"
    if has_rag_ingestions and any(k in message for k in RAG_TASKS):
        return "rag"
    if any(k in message for k in SUMMARY_TASKS):
        return "summarization"

    return "general"

def should_use_tools(state: AgentState) -> str:
    """
    This function determines whether to use tools defined in under
    'agent/tools'. It does so by checking if the last message in the
    graph state contains 'tool_calls' attribute.
    """
    last_message = state.messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END
