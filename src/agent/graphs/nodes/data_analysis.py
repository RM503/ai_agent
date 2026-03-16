from __future__ import annotations

from pathlib import Path

from langchain_core.messages import SystemMessage

from agent.prompts.load_prompts import load_prompts
from agent.schemas.graph_state import AgentState
from agent.services.llm import get_chat_model
from agent.tools.data_analysis.file_loader import file_loader
from agent.tools.data_analysis.python_repl import python_repl

PROMPTS_PATH = Path(__file__).resolve().parents[2] / "prompts" / "prompts.yaml"
DA_SYSTEM_PROMPT = load_prompts(name="data_analysis", category="system", prompts_file=PROMPTS_PATH)

def data_analysis_node(state: AgentState) -> dict:
    """
    This is the main entry point for the data analysis node.
    """
    llm = get_chat_model()

    # Context for uploaded files
    artifact_context = ""
    if state.uploaded_artifacts and state.uploaded_artifacts.file_path:
        file_path = state.uploaded_artifacts.file_path
        artifact_context = f"""
        Available uploaded file:
        file_path: {file_path}
        
        Use the `file_loader` tool with this exact file_path before analysis.
        """

    messages = [
        SystemMessage(DA_SYSTEM_PROMPT + artifact_context),
        *state.messages
    ]

    data_tools = [file_loader, python_repl]
    llm_with_tools = llm.bind_tools(data_tools)
    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response],
        "response_text": response.content if isinstance(response.content, str) else ""
    }