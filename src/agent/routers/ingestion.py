import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Form, File, UploadFile

from agent.common.logging_config import get_logger
from agent.ingestion.pipeline import load_and_split
from agent.memory.redis_config import redis_cache as r
from agent.vector_stores.factory import create_postgres_vector_store

logger = get_logger(__name__)

router: APIRouter = APIRouter(prefix="/ingestions", tags=["ingestions"])

STORAGE_DIR: Path = Path(__file__).resolve().parents[3] / "storage" / "ingestion"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/")
async def ingest_files(
    session_id: str | None = Form(...),
    user_id: str | None = Form(None),
    files: list[UploadFile] = File(...)
) -> dict[str, Any]:
    """
    This is the route for RAG pipeline file ingestion. It is different
    from the general upload route.
    """
    ingestion_id = uuid4().hex
    saved_paths: list[Path] = []
    results: dict[str, Any] = {}

    for file in files:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in [".pdf", ".txt", ".md", ".markdown", ".json", ".docx"]:
            results[file.filename] = {
                "status": "failed",
                "error": f"Unsupported file type: {suffix}"
            }
            continue
        file_id = uuid4().hex
        file_path = STORAGE_DIR / f"{file_id}_{suffix}"

        try:
            file_path.write_bytes(await file.read())
            saved_paths.append(file_path)
            results[file.filename] = {
                "status": "success",
                "file_id": file_id,
                "file_path": str(file_path)
            }
        except OSError as e:
            logger.exception(e)
            results[file.filename] = {
                "status": "failed",
                "error": str(e)
            }

    if not saved_paths:
        return {
            "ingestion_id": ingestion_id,
            "session_id": session_id,
            "results": results,
            "chunks_added": 0
        }

    chunks = load_and_split(saved_paths)
    for chunk in chunks:
        chunk.metadata.update({
            "ingestion_id": ingestion_id,
            "session_id": session_id,
            "user_id": user_id
        })

    ids = [uuid4().hex for _ in chunks]
    vector_store = create_postgres_vector_store()
    vector_store.add_documents(chunks, ids=ids)

    await r.set(
        f"session_id:{session_id}:ingesiont:{ingestion_id}",
        json.dumps({
            "ingestion_id": ingestion_id,
            "session_id": session_id,
            "user_id": user_id,
            "file_count": len(saved_paths),
            "chunk_count": len(chunks),
            "document_ids": ids
        })
    )

    return {
        "ingestion_id": ingestion_id,
        "session_id": session_id,
        "files_saved": len(saved_paths),
        "chunks_added": len(chunks),
        "document_ids": ids,
        "results": results
    }
