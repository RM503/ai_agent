"""
Module for RAG pipeline context builder.
"""

from __future__ import annotations

from agent.schemas.graph_state import RetrievedDocument


def build_citations(documents: list[RetrievedDocument]) -> list[dict[str, str]]:
    """
    Uses the list of documents retrieved by vector search and builds
    citation.

    Args:
        documents (list[RetrievedDocument]): List of documents retrieved by vector search.

    Returns:
        (list[dict[str, sr]]): List of dictionaries, each containining a citation for the
            chunk retrieved.
    """
    citations: list[dict[str, str]] = []

    for idx, document in enumerate(documents, start=1):
        citations.append(
            {
                "citation_id": idx,
                "source": document.source,
                "file_name": document.file_name,
                "page": document.page,
                "chunk_idx": document.chunk_idx,
                "doc_id": document.doc_id if document.doc_id else None,
                "ingestion_id": document.ingestion_id if document.ingestion_id else None
            }
        )

    return citations


def build_context_text(documents: list[RetrievedDocument]) -> str:
    """
    Builds a block of context text from documents retrieved by vector search.
    """
    blocks = []

    for idx, document in enumerate(documents, start=1):
        label_parts = [f"Source {idx}"]

        if document.file_name:
            label_parts.append(f"file={document.file_name}")
        else:
            label_parts.append(f"source={document.source}")

        if document.page is not None:
            label_parts.append(f"chunk={document.chunk_idx}")

        if document.chunk_idx is not None:
            label_parts.append(f"chunk={document.chunk_idx}")

        label = ", ".join(label_parts)
        blocks.append(f"[{label}]\n{document.content.strip()}")

    return "\n\n".join(blocks)
