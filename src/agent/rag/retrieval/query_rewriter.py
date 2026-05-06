"""
Modulue for rewriting user queries for RAG pipeline
"""

from __future__ import annotations

import json

from json import JSONDecodeError
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from agent.prompts.load_prompts import load_prompts
from agent.schemas.graph_state import RagQuery
from agent.services.llm import get_chat_model

PROMPTS_PATH = Path(__file__).resolve().parents[3] / "prompts" / "prompts.yaml"


class QueryRewriteInput(BaseModel):
    """
    Class for input for query rewrite.

    Attributes:
        user_query (str): User's query for RAG
        recent_messages (list[BaseMessage]): A list of recent messages in the chat
        conversation_summary (str | None): A summary of recent conversation history;
            defaults to None
        active_ingestion_id (str | None): ID of ingestion documents, if happened in session;
            defaults to None
        filters (dict[str, Any]): Additional key-word based filters
    """
    user_query: str
    recent_messages: list[BaseMessage] = Field(default_factory=list)
    conversation_summary: str | None = None
    active_ingestion_id: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)


def _fallback_query(payload: QueryRewriteInput) -> RagQuery:
    return RagQuery(
        original_query=payload.user_query,
        rewritten_query=payload.user_query,
        needs_retrieval=True,
        query_type="qa",
        filters=payload.filters
    )


def summarize_history(payload: QueryRewriteInput) -> dict[str, str]:
    """
    Summarizes conversation history leading to current user query. Ignores current
    user query and constructs a summary from recent conversation history if certain
    conditions are satisfied.

    Args:
        payload (QueryRewriteInput): Query rewrite input payload.

    Returns (dict[str, Any]): Summary of recent conversation.
    """
    recent_messages = payload.recent_messages
    min_message_length_for_summary = 4
    message_window_for_summary = 10

    if not recent_messages or len(recent_messages) < min_message_length_for_summary:
        return {"conversation_summary": ""}

    # Retrieves relevant AI and Human messages, excluding tool calls, for summarization
    relevant_messages = [
        message for message in recent_messages
        if isinstance(message, AIMessage | HumanMessage)
        and not getattr(message, "tool_calls", None)
    ]
    if not relevant_messages:
        return {"conversation_summary": ""}

    # Construct a conversation from recent history such that user and AI
    # roles are specified in string
    conversation = "Conversation history:\n"
    for message in relevant_messages[-message_window_for_summary:]: # Limit to most recent messages
        role = "User" if isinstance(message, HumanMessage) else "AI"
        conversation += f"{role}: {message.content}\n"

    system_prompts = load_prompts(
        name="conversation_summarization",
        category="system",
        prompts_file=PROMPTS_PATH
    )
    llm = get_chat_model()
    chain = llm | StrOutputParser()

    summary_response = chain.invoke(
        [
            SystemMessage(content=system_prompts),
            HumanMessage(content=conversation)

        ]
    )
    return {"conversation_summary": str(summary_response.content).strip()}


def rewrite_query(payload: QueryRewriteInput) -> RagQuery:
    """
    Rewrites current user query based on conversation summary (if present).
    """
    if not payload.user_query.strip():
        return RagQuery(
            original_query=payload.user_query,
            rewritten_query="",
            needs_retrieval=False,
            query_type="other",
            filters=payload.filters
        )

    conversation_summary = payload.conversation_summary
    if conversation_summary is None:
        conversation_summary = summarize_history(payload).get("conversation_summary")

    system_prompt = load_prompts(
        name="rewrite_query",
        type="system",
        prompts_file=PROMPTS_PATH
    )
    user_query = f"""
    Conversation summary:
    {conversation_summary or "No conversation summary available."}

    Latest user query:
    {payload.user_query}
    """.strip()

    llm = get_chat_model()
    chain = llm | StrOutputParser()

    try:
        response = chain.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_query)
            ]
        )
        parsed = json.loads(str(response.content).strip())
    except (JSONDecodeError, TypeError, ValueError):
        return _fallback_query(payload)

    query_type = parsed.get("query_type")
    if query_type not in {"qa", "summary", "comparison", "lookup", "other"}:
        # Fallback to QA if query_type does not exist
        query_type = "qa"

    rewritten_query = str(parsed.get("rewritten_query") or payload.user_query).strip()

    return RagQuery(
        original_query=payload.user_query,
        rewritten_query=rewritten_query,
        needs_retrieval=bool(parsed.get("needs_retrieval", True)),
        query_type=query_type,
        filters=payload.filters
    )
