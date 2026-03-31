from __future__ import annotations

from pathlib import Path

from langchain_core.messages import SystemMessage

from .utils import get_recent_messages, trim_tool_message
from agent.prompts.load_prompts import load_prompts
from agent.schemas.graph_state import AgentState
from agent.services.llm import get_chat_model
from agent.tools.weather import get_weather
from agent.tools.web_search import web_search

PROMPTS_PATH = Path(__file__).resolve().parents[2] / "prompts" / "prompts.yaml"
GENERAL_SYSTEM_PROMPT = load_prompts(name="general", category="system", prompts_file=PROMPTS_PATH)

def general_responder_node(state: AgentState) -> dict:
    """
    This is the general responder node for chatting and basic
    tool usage.
    """
    llm = get_chat_model() # use simpler model for general node

    messages = [
        SystemMessage(GENERAL_SYSTEM_PROMPT),
        *trim_tool_message(get_recent_messages(state.messages))
    ]

    tools = [get_weather, web_search]
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response],
        "response_text": response.content
    }

