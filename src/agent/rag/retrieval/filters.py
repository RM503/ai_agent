from __future__ import annotations

from typing import Any
from uuid import UUID

from agent.schemas.graph_state import RagScope


def _stringify(value: str | UUID | None) -> str | None:
    if value is None:
        return None
    return str(value)

def build_rag_filters(
        rag_scope: RagScope,
        session_id: str | UUID | None = None
) -> dict[str, Any]:
    filters: dict[str, Any] = {}

    if rag_scope.scope == "global":
        pass

    elif rag_scope.scope == "user":
        user_id = _stringify(rag_scope.user_id)
        if user_id:
            filters["user_id"] = user_id

    elif rag_scope.scope == "project":
        project_id = _stringify(rag_scope.project_id)
        if project_id:
            filters["project_id"] = project_id

    elif rag_scope.scope == "session":
        scoped_session_id = _stringify(session_id)
        if scoped_session_id:
            filters["session_id"] = scoped_session_id

    if rag_scope.collection:
        filters["collection"] = rag_scope.collection

    ingestion_id = _stringify(rag_scope.ingestion_id)
    if ingestion_id:
        filters["ingestion_id"] = ingestion_id

    return filters
