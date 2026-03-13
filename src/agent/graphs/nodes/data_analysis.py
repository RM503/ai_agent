from __future__ import annotations

from pathlib import Path

from langchain_core.messages import SystemMessage

from agent.prompts.load_prompts import load_prompts
from agent.schemas.graph_state import AgentState
from agent.services.llm import get_chat_model
from agent.tools.data_analysis.file_loader import file_loader
from agent.tools.data_analysis.parse_inline_data import register_inline_dataset
from agent.tools.data_analysis.pythonrepl import python_repl

PROMPTS_PATH = Path(__file__).resolve().parents[2] / "prompts" / "prompts.yaml"
DA_SYSTEM_PROMPT = load_prompts(name="data_analysis", category="system", prompts_file=PROMPTS_PATH)

def _extract_inline_data_candidates(state: AgentState) -> str:
    for message in reversed(state.messages):
        if getattr(message, "type", None) == "human":
            return str(message.content)
    return ""

def data_analysis_node(state: AgentState) -> dict:
    """
    This is the main entry point for the data analysis node.
    """
    llm = get_chat_model()

    dataset_key = state.dataset_key

    messages = [
        SystemMessage(DA_SYSTEM_PROMPT),
        *state.messages
    ]

    data_tools = [file_loader, python_repl]
    llm_with_tools = llm.bind_tools(data_tools)
    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response],
        "response_text": response.content
    }