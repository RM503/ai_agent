from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .utils import get_recent_messages
from agent.common.logging_config import get_logger
from agent.prompts.load_prompts import load_prompts
from agent.rag.retrieval.context_builder import build_context_text
from agent.rag.retrieval.filters import build_rag_filters
from agent.rag.retrieval.query_rewriter import QueryRewriteInput, rewrite_query
from agent.rag.retrieval.retriever import retrieve_for_query
from agent.schemas.graph_state import AgentState, RagContext
from agent.services.llm import get_chat_model

logger = get_logger(__name__)


PROMPTS_PATH = Path(__file__).resolve().parents[2] / "prompts" / "prompts.yaml"
RAG_SYSTEM_PROMPT = load_prompts(
    name="rag", category="system", prompts_file=PROMPTS_PATH
)


def _last_user_message(state: AgentState) -> str:
    """Retrieve last user message."""
    for message in reversed(state.messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _answer_from_context(
        state: AgentState,
        rag_context: RagContext
) -> str:
    query = rag_context.query
    original_query = query.original_query if query else _last_user_message(state)
    retrieval_status = rag_context.retrieval_status

    if retrieval_status == "empty":
        return "I could not find enough relevant information in the ingested text to answer that."

    if retrieval_status == "error":
        return f"I could not retrieve the ingested documents due to an error: {rag_context.error}"

    context_text = build_context_text(rag_context.documents)

    messages = [
        SystemMessage(content=RAG_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"User question:\n{original_query}\n\n"
                f"Retrieved context:\n{context_text}"
            )
        )
    ]

    llm = get_chat_model()
    response = llm.invoke(messages)
    return str(response.content)


def rag_node(state: AgentState) -> dict[str, Any]:
    user_query = _last_user_message(state)

    # Create metadata filters
    filters = build_rag_filters(
        session_id=state.session_id,
        ingestion_id=state.ingestion_id
    )

    # Rewrite query
    rewrite_input = QueryRewriteInput(
        user_query=user_query,
        recent_messages=get_recent_messages(state.messages, max_turns=3),
        conversation_summary=state.summary_text,
        active_ingestion_id=str(state.active_ingestion_id) if state.active_ingestion_id else None,
        filters=filters
    )

    rag_query = rewrite_query(rewrite_input)
    rag_context = retrieve_for_query(rag_query)

    answer = _answer_from_context(state, rag_context)

    return {
        "messages": [AIMessage(content=answer)],
        "rag_context": rag_context
    }
