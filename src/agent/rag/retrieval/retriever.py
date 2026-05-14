"""
Main retriever module for RAG pipeline
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from .context_builder import build_citations
from agent.rag.postprocessing.dedupe import dedupe_documents
from agent.schemas.graph_state import (
    RagContext,
    RagQuery,
    RetrievedDocument
)
from agent.vector_stores.factory import create_qdrant_vector_store


def _metadata_value(metadata: dict[str, Any], *keys: str) -> Any | None:
    """Retrieves metadata from document for a given key."""
    for key in keys:
        if key in metadata:
            return metadata[key]
    return None


def _add_score_to_metadata(
        docs: list[Document],
        scores: list[float]
) -> list[Document]:
    """Adds vector search scores to document metadata."""
    for doc, score in zip(docs, scores):
        doc.metadata["score"] = score
    return docs


def _to_retrieved_document(document: Document) -> RetrievedDocument:
    """Converts document chunk into RetrievedDocument object."""
    metadata = dict(document.metadata or {})

    return RetrievedDocument(
        content=document.page_content,
        source=_metadata_value(metadata, "source"),
        doc_id=_metadata_value(metadata, "_id", "id", "doc_id"),
        ingestion_id=_metadata_value(metadata, "ingestion_id"),
        file_name=_metadata_value(metadata, "file_name", "filename"),
        page=_metadata_value(metadata, "page", "page_number"),
        chunk_idx=_metadata_value(metadata, "chunk_idx"),
        score=_metadata_value(metadata, "score"),
        metadata=metadata,
    )


def _passses_score_threshold(
        doc: RetrievedDocument,
        score_threshold: float | None
) -> bool:
    if score_threshold is None:
        return True
    if doc.score is None:
        return True
    return doc.score >= score_threshold


def retrieve_for_query(
        query: RagQuery,
        *,
        k: int = 12,
        score_threshold: float = 0.5,
        final_k: int = 6
) -> RagContext:
    """
    Main retriever function for RAG pipeline. This uses rewritten or original query
    from RagQuery, creates vector store, performs similarity search and document
    deduplication.

    Args:
        query (RagQuery): The RagQuery object
        k (int): Number of nearest neighbor documents to be retrieved; defaults to 12
        score_threshold (float): Minimum similarity score needed for a document to be
            in retrieved documents list; defaults to 0.5
        final_k (int): The number of documents to be passed to RagContext.
    """
    if not query.needs_retrieval:
        return RagContext(query=query, retrieval_status="not_run")

    # Use rewritten query if not None
    search_query = query.rewritten_query or query.original_query

    try:
        vector_store = create_qdrant_vector_store()
        docs, scores = vector_store.similarity_search_with_score(
            query=search_query,
            k=k,
            filter=query.filters or None,
            score_threshold=score_threshold
        )
    except Exception as e:
        return RagContext(
            query=query,
            retrieval_status="error",
            error=str(e)
        )

    # Include retrieval scores as part of document metadata
    docs = _add_score_to_metadata(docs, scores)

    retrieved = [_to_retrieved_document(doc) for doc in docs]
    retrieved = [
        doc for doc in retrieved
        if _passses_score_threshold(doc, score_threshold)
    ] # Reduce list of documents to ones which meet score threshold

    # Deduplicate and sort by score in descending order
    retrieved_deduped = dedupe_documents(retrieved)[:final_k]
    retrieved_sorted = sorted(
        retrieved_deduped,
        key=lambda doc: doc.score if doc.score is not None else float("-inf"),
        reverse=True
    )

    if not retrieved:
        return RagContext(
            query=query,
            documents=[],
            citations=[],
            retrieval_status="empty"
        )

    return RagContext(
        query=query,
        documents=retrieved_sorted,
        citations=build_citations(retrieved),
        retrieval_status="found"
    )
