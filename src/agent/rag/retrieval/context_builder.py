from __future__ import annotations

from agent.schemas.graph_state import RetrievedDocument


def build_citations(documents: list[RetrievedDocument]) -> list[dict[str, str]]:
    citations: list[dict[str, str]] = []

    for idx, document in enumerate(documents, start=1):
        citations.append(
            {
                "citation_id": idx,
                "source": document.source,
                "file_name": document.file_name,
                "page": document.page,
                "chunk_idx": document.chunk_idx,
                "doc_id": document.doc_id,
                "ingestion_id": document.ingestion_id
            }
        )

    return citations


def build_context_text(documents: list[RetrievedDocument]) -> str:
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
