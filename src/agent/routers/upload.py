import json
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Form, File, UploadFile
from redis.asyncio import Redis

from agent.common.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()

r = Redis(host="localhost", port=6379, db=3, decode_responses=True)
STORAGE_DIR = Path(__file__).resolve().parents[3] / "storage" / "uploads"
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
        content = await file.read()
        file_id = str(uuid4().hex)
        file_name = file.filename
        file_path = STORAGE_DIR / file.filename

        payload = {
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
        key = f"session_id:{session_id}:file"

        await r.set(key, json.dumps(payload))

        await r.expire(key, 3600)

    return {"results": results}
