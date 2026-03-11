# Routes messages to appropriate agents; implements the `decide route` function
from __future__ import annotations

from pathlib import Path

from langgraph.graph import END

from agent.schemas.graph_state import AgentState

# Set of audio and data file extensions
AUDIO_EXTS: set[str] = {".mp3", ".wav", ".m4a", ".mp4", ".aac", ".flac"}
DATA_EXTS: set[str] = {".csv", ".xlsx", ".xls", ".parquet"}

# Possible tasks list
TRANSCRIPTION_TASKS: list[str] = ["transcribe", "transcription", "audio", "meeting recording"]
ANALYSIS_TASKS: list[str] = ["analyze dataset", "analyze csv", "data analysis", "summarize dataframe", "profile this data"]
SUMMARY_TASKS: list[str] = ["summarize", "meeting notes", "minutes", "summary"]

def decide_route(message: str, file_name: str | None=None) -> str:
    """
    This function implements a router for the orchestrator for delegating
    tasks to sub-agents from message contexts. As of now, the context might be
    narrow.

    Args:
        message (str): the message to be processed.
        file_name (str, optional): name of file (if passed)
    Returns:
        str: the route identifier.
    """
    message = message.lower()
    suffix = Path(file_name).suffix.lower() if file_name else ""

    if suffix in AUDIO_EXTS or any(k in message for k in TRANSCRIPTION_TASKS):
        return "transcription"
    if  suffix in DATA_EXTS or any(k in message for k in ANALYSIS_TASKS):
        return "data_analysis"
    if any(k in message for k in SUMMARY_TASKS):
        return "summary"
    else:
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