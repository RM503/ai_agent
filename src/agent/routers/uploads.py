import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.templating import Jinja2Templates

from agent.core.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# Resolve root and find uploads directory
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
async def upload_audio(file: UploadFile = File(...)):
    # Check if data is indeed audio
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio type.")

    file_path = UPLOAD_DIR / file.filename

    logger.info(f"Saving to {file_path}")

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "saved_to": str(file_path)
    }