import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Form, File, HTTPException, UploadFile
from kombu.exceptions import OperationalError

from agent.common.logging_config import get_logger
from agent.memory.redis_config import redis_cache as r
from agent.worker.tasks import ingest_documents

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

    Args:
        session_id (str | None): The session ID to associate the ingestion with.
        user_id (str | None): The user ID to associate the ingestion with.
        files (list[UploadFile]): The list of files to ingest.

    Returns:
        dict[str, Any]: A dictionary containing the ingestion ID, session ID, status, number
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

    # Trigger background ingestion task
    await r.set(f"ingestion:{ingestion_id}:status", "queued")
    await r.delete(f"ingestion:{ingestion_id}:error")
    await r.delete(f"ingestion:{ingestion_id}:result")

    try:
        ingest_documents.delay(
            ingestion_id=ingestion_id,
            session_id=session_id,
            user_id=user_id,
            file_paths=[str(path) for path in saved_paths],
            initial_results=results
        )
    except OperationalError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to enqueue ingestion task: {str(e)}"
        )

    return {
        "ingestion_id": ingestion_id,
        "session_id": session_id,
        "status": "queued",
        "files_saved": len(saved_paths),
        "status_url": f"/ingestions/status/{ingestion_id}",
        "results": results
    }

@router.get("/status/{ingestion_id}")
async def get_ingestion_status(ingestion_id: str) -> dict[str, Any]:
    status = await r.get(f"ingestion:{ingestion_id}:status") or "unknown"

    payload = {
        "ingestion_id": ingestion_id,
        "status": status
    }

    if status == "error":
        payload["error"] = await r.get(f"ingestion:{ingestion_id}:error")

    if status == "done":
        raw_result = await r.get(f"ingestion:{ingestion_id}:result")
        if raw_result:
            payload["result"] = json.loads(raw_result)

    return payload
