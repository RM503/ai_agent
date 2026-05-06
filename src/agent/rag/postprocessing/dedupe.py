from __future__ import annotations

from agent.schemas.graph_state import RetrievedDocument

def dedupe_documents(documents: list[RetrievedDocument]) -> list[RetrievedDocument]:
    """Deduplicates retrieved documents for RAG."""
    seen: set[str] = set()
    deduped: list[RetrievedDocument] = []

    for document in documents:
        key = "|".join(
            [
                str(document.source or ""),
                str(document.ingestion_id or ""),
                str(document.chunk_idx or ""),
                document.content.strip()[:200]
            ]
        )

        if key in seen:
            continue

        seen.add(key)
        deduped.append(document)

    return deduped
