# Routers for file uploads

from pathlib import Path
from typing import Any
from uuid import uuid4

from celery.result import AsyncResult
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.templating import Jinja2Templates

from agent.core.logging_config import get_logger
from agent.worker.celery_app import celery_app
from agent.worker.tasks import transcribe_audio

logger = get_logger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory="agent/templates")

# Resolve root and find uploads directory
# All uploads are stored at the root level under storage/uploads
ROOT_DIR = Path(__file__).resolve().parents[3]
UPLOAD_DIR = ROOT_DIR / "storage" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.get("/upload")
async def upload_page(request: Request):
    return templates.TemplateResponse(
        "upload.html",
        {"request": request}
    )

@router.post("/upload-audio", tags=["upload-audio"])
async def upload_audio_for_transcription(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    This route allows users to upload audio files for transcription/translation.
    """
    # Check if data is indeed audio
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio type.")

    # Save file
    suffix = Path(file.filename).suffix or "bin"
    file_id = uuid4().hex
    file_path = UPLOAD_DIR / f"{file_id}{suffix}"

    data = await file.read()
    logger.info(f"Saving to {file_path}")
    file_path.write_bytes(data)

    # Enqueue Celery job
    task = transcribe_audio.delay(str(file_path))

    return {
        "job_id": task.id,
        "file_id": file_id,
        "status_url": f"/transcription/status/{task.id}",
        "result_url": f"/transcription/result/{task.id}"
    }

@router.get("/status/{job_id}")
def status(job_id: str) -> dict[str, Any]:
    r = AsyncResult(job_id, app=celery_app)
    payload = {"job_id": job_id, "state": r.state}

    if r.state == "FAILURE":
        payload["error"] = str(r.result)
    return payload

@router.get("/status/{job_id}")
def result(job_id: str) -> dict[str, Any]:
    r = AsyncResult(job_id, app=celery_app)

    if r.state in ("PENDING", "STARTED", "RETRY"):
        raise HTTPException(status_code=202, detail=f"Not ready. State={r.state}")
    if r.state == "FAILURE":
        raise HTTPException(status_code=500, detail=str(r.result))

    # if SUCCESS
    return {"job_id": job_id, "result": r.result}