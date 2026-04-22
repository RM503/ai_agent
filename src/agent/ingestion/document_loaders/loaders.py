from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from .registry import register_default_loader, register_loader

def _metadata(path: Path, **extra: Any) -> dict[str, Any]:
    return {
        "source": str(path),
        "file_name": path.name,
        "extension": path.suffix.lower(),
        **extra,
    }

def _read_text(path: Path, options: dict[str, Any] | None = None) -> str:
    options = options or {}
    return path.read_text(
        encoding=options.get("encoding", "utf-8"),
        errors=options.get("errors", "replace"),
    )

@register_default_loader
@register_loader(".txt", ".md", ".markdown")
def load_text_document(path: Path, options: dict[str, Any] | None = None) -> list[Document]:
    return [
        Document(
            page_content=_read_text(path, options),
            metadata=_metadata(path),
        )
    ]

@register_loader(".json")
def load_json_document(path: Path, options: dict[str, Any] | None = None) -> list[Document]:
    options = options or {}
    with path.open("r", encoding=options.get("encoding", "utf-8")) as file:
        payload = json.load(file)

    return [
        Document(
            page_content=json.dumps(payload, indent=2, ensure_ascii=False),
            metadata=_metadata(path),
        )
    ]

@register_loader(".pdf")
def load_pdf_document(path: Path, options: dict[str, Any] | None = None) -> list[Document]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    documents: list[Document] = []

    for page_number, page in enumerate(reader.pages, start=1):
        documents.append(
            Document(
                page_content=page.extract_text() or "",
                metadata=_metadata(path, page_number=page_number),
            )
        )

    return documents

@register_loader(".docx")
def load_docx_document(path: Path, options: dict[str, Any] | None = None) -> list[Document]:
    import docx2txt

    return [
        Document(
            page_content=docx2txt.process(str(path)) or "",
            metadata=_metadata(path),
        )
    ]
