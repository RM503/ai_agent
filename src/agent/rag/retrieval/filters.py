from __future__ import annotations

from typing import Any
from uuid import UUID


def build_rag_filters(
        session_id: str | UUID,
        user_id: str | UUID | None = None,
        ingestion_id: str | UUID | None = None
) -> dict[str, Any]:
    filters = {"session_id": str(session_id)}

    if user_id:
        filters["user_id"] = str(user_id)

    if ingestion_id:
        filters["ingestion_id"] = str(ingestion_id)

    return filters
