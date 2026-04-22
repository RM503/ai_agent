"""
Module containing data ingestion pipeline for RAG.
"""
from __future__ import annotations 

from pathlib import Path
from typing import Any, Iterable, Optional

from langchain_core.documents import Document

from .document_loaders.registry import load_documents
from .text_splitters.text_splitters import TextSplitter

def _normalize_paths(paths: str | Path | list[str | Path]) -> list[Path]:
    """
    Normalizes document paths for single and multiple document sources by returning
    an appropriate list of path(s).

    Args:
        paths (str | Path | list[str | Path]): single or list of paths in str or Path format

    Returns:
        (list[Path]): a list of normalied path object(s)
    """
    if isinstance(paths, (str, Path)):
        normalized = [Path(paths)]
    else:
        normalized = [Path(p) for p in paths]
    
    if not normalized:
        raise ValueError("No input paths provided.")

    return normalized

def _load_all(paths: list[Path], **loader_kwargs: dict[str, Any]) -> list[Document]:
    """
    Loads all documents for specified path(s).

    Args:
        paths (list[Path]): a list of path(s) to document(s)

    Returns:
        (list[Document]): a list of LangChain `Document` objects with `page_content` and `metadata`
    """
    documents: list[Document] = []

    for path in paths:
        documents.extend(load_documents(path, **loader_kwargs))

    if not documents:
        raise ValueError("No documents were loaded.")

    return documents

def _attach_chunk_metadata(chunks: list[Document]) -> list[Document]:
    """
    Attaches source specific `chunk_idx` to metadata. For eg.

    source_1 --> chunk_idx=0, chunk_idx=1, ...
    source_2 --> chunk)idx=0, chunk_idx=2, ...
    ...

    Args:
        chunks (list[Documents]): a list of all document chunks with metadata

    Returns:
        (list[Document]): same list of chunks with additional `chunk_idx` metadata
    """
    counts_by_source: dict[str, int] = {}

    for chunk in chunks:
        source = str(chunk.metadata.get("source", "unknown"))
        chunk_idx = counts_by_source.get(source, 0)
        chunk.metadata["chunk_idx"] = chunk_idx 
        counts_by_source[source] = chunk_idx + 1

    return chunks

def load_and_split(
    paths: str | Path | Iterable[str | Path],
    splitter: Optional[TextSplitter]=None,
    loader_kwargs: Optional[dict[str, Any]]=None,
    splitter_kwargs: Optional[dict[str, Any]]=None
) -> list[Document]:
    """
    Loads all documents and performs split.

    Args:
        paths (str | Path | Iterable[str | Path]): document path(s)
        splitter (TextSplitter, optional): text-splitter class 
        loader_kwargs (dict[str, Any], optional): key-word arguments passed to loader
        splitter_kwargs (dict[str, Any], optional): key-word arguments passed to splitter class

    Returns:
        (list[Document]): list of chunked document(s)
    """
    normalized_paths = _normalize_paths(paths)
    loaded_documents = _load_all(normalized_paths, **loader_kwargs)

    splitter = splitter or TextSplitter(**splitter_kwargs)
    chunked_documents = splitter.create_splitter().split_documents(loaded_documents)

    return _attach_chunk_metadata(chunked_documents)