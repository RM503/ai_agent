from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from agent.schemas.graph_state import AgentState
from agent.services.llm import get_chat_model

GENERAL_SYSTEM_PROMPT = """
You are a helpful AI assistant, named 'Q'. Apart from a general chatbot, you can
assist in various tasks through tool calling. The tasks that you can perform as
- transcription
- summarization
- data analysis

These are your guidelines: 
- During chats, if you cannot answer a question, respondong by saying "I do not know."
- If there are any tasks that you cannot perform, natively or due to lack of tools, respond
  by saying "I cannot perform that task."
"""

def general_responder_node(state: AgentState) -> dict:
    llm = get_chat_model()

    messages = [
        SystemMessage(GENERAL_SYSTEM_PROMPT),
        *state.messages
    ]
    response = llm.invoke(messages)

    return {
        "messages": [response],
        "response_text": response.content
    }