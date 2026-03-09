# Routers for file uploads and transcriptions

import asyncio
import json
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from kombu.exceptions import OperationalError
from redis import Redis

from agent.common.logging_config import get_logger
from agent.worker.tasks import transcribe_audio

logger = get_logger(__name__)

page_router = APIRouter()
api_router = APIRouter(prefix="/transcription", tags=["transcription"])
templates = Jinja2Templates(directory="agent/templates")

# Resolve root and find uploads directory
# All uploads are stored at the root level under storage/uploads
ROOT_DIR = Path(__file__).resolve().parents[3]
UPLOAD_DIR = ROOT_DIR / "storage" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Initialize Redis client
r = Redis(host="localhost", port=6379, db=0, decode_responses=True)
r.config_set("save", "")

@page_router.get("/upload")
async def upload_page(request: Request) -> HTMLResponse:
    """Get the upload page."""
    return templates.TemplateResponse(
        "upload.html",
        {"request": request}
    )

@api_router.post("/upload-audio", tags=["upload-audio"])
async def upload_audio_for_transcription(file: UploadFile = File(...)) -> dict[str, Any]:
    """This route allows users to upload audio files for transcription/translation."""
    # Check if data is indeed audio
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio type.")

    # Public Redis job_id
    job_id = uuid4().hex

    # Save file
    suffix = Path(file.filename).suffix
    file_path = UPLOAD_DIR / f"{job_id}{suffix}"

    data = await file.read()
    logger.info(f"Saving to {file_path}")
    file_path.write_bytes(data)

    # Initialize Redis state; Celery updates 'status' from queued to 'done' or 'error'
    r.set(f"stt:{job_id}:status", "queued")
    r.delete(f"stt:{job_id}:events")
    r.delete(f"stt:{job_id}:error")
    r.delete(f"stt:{job_id}:result_path")

    # Enqueue Celery job
    try:
        transcribe_audio.delay(job_id, str(file_path))
    except OperationalError as e:
        raise HTTPException(status_code=503, detail=f"Celery broker unavailable: {e}")

    return {
        "job_id": job_id,
        "stream_url": f"/transcription/stream/{job_id}",
        "status_url": f"/transcription/status/{job_id}",
        "download_url": f"/transcription/download/{job_id}"
    }

@api_router.get("/status/{job_id}")
def status(job_id: str) -> dict[str, Any]:
    status = r.get(f"stt:{job_id}:status") or "unknown"
    payload = {"job_id": job_id, "status": status}

    if status == "error":
        payload["error"] = r.get(f"stt:{job_id}:error")
    if status == "done":
        payload["result_path"] = r.get(f"stt:{job_id}:result_path")

    return payload

@api_router.get("/stream/{job_id}")
async def stream(job_id: str) -> StreamingResponse:
    """
    SSE streaming endpoint.
        - Reads JSON events from Redis list stt:{job_id}:events
        - Emits them as SSE 'data: ...'
    """
    key = f"stt:{job_id}:events"
    async def event_generator() -> AsyncIterator[str]:
        idx = 0
        while True:
            n = r.llen(key)
            while idx < n:
                raw = r.lindex(key, idx)
                idx += 1
                if raw is None:
                    continue

                # raw is already in JSON format
                yield f"data: {raw}\n\n"

                # Stop on done/error events
                try:
                    msg = json.loads(raw)
                    if msg.get("type") in ("done", "error"):
                        return

                except Exception:
                    pass

            await asyncio.sleep(0.25)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@api_router.get("/download/{job_id}")
def download(job_id: str) -> FileResponse:
    status = r.get(f"stt:{job_id}:status")
    if status != "done":
        raise HTTPException(status_code=409, detail=f"Not ready (status={status})")

    path = r.get(f"stt:{job_id}:result_path")
    if not path:
        raise HTTPException(status_code=404, detail=f"Missing result_path in Redis")

    path = Path(path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Transcript file missing on disk")

    return FileResponse(str(path), media_type="text/plain", filename=path.name)