from __future__ import annotations

import json
from json.decoder import JSONDecodeError
from pathlib import Path

from langchain_core.messages import SystemMessage

from .utils import get_recent_messages
from agent.common.logging_config import get_logger
from agent.prompts.load_prompts import load_prompts
from agent.schemas.graph_state import AgentState
from agent.services.llm import get_chat_model

logger = get_logger(__name__)

PROMPTS_PATH = Path(__file__).resolve().parents[2] / "prompts" / "prompts.yaml"
ORCHESTRATOR_SYSTEM_PROMPT = load_prompts(name="orchestrator", category="system", prompts_file=PROMPTS_PATH)

# Valid routes and fallback route
VALID_ROUTES: set[str] = {"transcription", "data_analysis", "summarization", "general"}
FALLBACK_ROUTE = "general"

def orchestrator_node(state: AgentState) -> dict:
    """
    You are the 'Orchestrator node'. You decide which node to
    direct the state to depending on what the user wants using
    the `decide_route` function.
    """
    llm = get_chat_model()

    # Important: add upload context to orchestrator
    artifact_context = ""
    if state.uploaded_artifacts and state.uploaded_artifacts.file_path:
        artifact_context = f"\nThe user has uploaded a file: {state.uploaded_artifacts.file_name}. Route to data_analysis."

    messages = [
        SystemMessage(ORCHESTRATOR_SYSTEM_PROMPT + artifact_context),
        *get_recent_messages(state.messages)
    ]

    response = llm.invoke(messages)
    content = response.content.strip().lower()

    try:
        parsed_content = json.loads(content)
        route = parsed_content.get("route", "").strip().lower()
    except JSONDecodeError:
        route = content.lower()

    # General fallback if a valid route was not identified
    if route not in VALID_ROUTES:
        route = FALLBACK_ROUTE

    logger.info(f"Orchestrator routed to: {route}")
    return {"route": route}