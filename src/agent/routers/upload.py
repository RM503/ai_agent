import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Form, File, UploadFile

from agent.common.logging_config import get_logger
from agent.memory.redis_config import redis_cache as r

logger = get_logger(__name__)

router: APIRouter = APIRouter()

STORAGE_DIR: Path = Path(__file__).resolve().parents[3] / "storage" / "uploads"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/uploads")
async def upload_files(
        session_id: str | None = Form(...),
        user_id: str | None = Form(None),
        files: list[UploadFile] = File(...)
):
    """
    Route for file uploads.
    """
    results: dict[str, str] = {} # store file upload results
    for file in files:
        # for UploadArtifact state, except file content
        # which is to be read from storage
        content: bytes = await file.read()
        file_id: str = str(uuid4().hex)
        file_name: str = file.filename
        file_path: Path = STORAGE_DIR / file.filename

        payload: dict[str, Any] = {
            "session_id": session_id,
            "user_id": user_id,
            "file_id": file_id,
            "file_name": file_name,
            "file_path": str(file_path)
        }

        try:
            with open(file_path, "wb") as f:
                f.write(content)
            results[file_name] = "succes"
        except (FileNotFoundError, IOError) as e:
            logger.exception(e)
            results[file_name] = "failed"

            continue

        # store payload in Redis with appropriate tags
        # set TTL to 1 hr
        key: str = f"session_id:{session_id}:file"

        await r.set(key, json.dumps(payload))

        await r.expire(key, 3600)

    return {"results": results}
