# Celery task for transcription
from __future__ import annotations

import json

import redis
from pathlib import Path
from typing import Any

from .celery_app import celery_app
from ..tools.transcription.stt import transcription_stream

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "storage" / "transcripts"

r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
r.config_set("save", "")

def _push_event(job_id: str, payload: dict[str, Any]) -> None:
    r.rpush(f"stt:{job_id}:events", json.dumps(payload, ensure_ascii=False))

@celery_app.task(bind=True, name="task.transcribe_audio")
def transcribe_audio(self, job_id: str, file_path: str, out_dir: str=OUTPUT_DIR) -> dict[str, Any]:
    """
    This function creates a Celery task for audio transcription.

    Args:
        job_id (str) - job id for Redis queue
        file_path (str) - audio file path
        out_dir (str) - output directory for transcribed file

    Returns:
        dict[str, Any]
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Missing audio file: {file_path}")

        r.set(f"stt:{job_id}:status", "running")
        r.delete(f"stt:{job_id}:error")

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        txt_path = out_dir / f"{file_path.stem}.txt"

        final_text = None
        final_meta = {}
        # Generate transcription stream
        segments = []
        for event in transcription_stream(file_path):
            segments.append(event)

            _push_event(job_id, event)

            event_type = event.get("type")
            if event_type == "meta":
                final_meta = event
            elif event_type == "done":
                # When `done` event will yield the full text
                final_text = event.get("text", "") or ""
            elif event_type == "error":
                # If stream generator yields errors
                msg = event.get("message", "Unknown error")
                raise RuntimeError(f"Transcription error: {msg}")

        if final_text is None:
            raise RuntimeError(f"Transcription stream ended without a `done` event.")

        txt_path.write_text(final_text, encoding="utf-8")

        r.set(f"stt:{job_id}:result_path", str(txt_path))
        r.set(f"stt:{job_id}:status", "done")

        return {
            "job_id": job_id,
            "status": "done",
            "text_path": str(txt_path),
            "language": final_meta.get("language"),
            "duration_s": final_meta.get("duration_s")
        }

    except Exception as e:
        r.set(f"stt:{job_id}:status", "error")
        r.set(f"stt:{job_id}:error", str(e))

        # Produce an error message at the frontend
        _push_event(job_id, {"type": "error", "message": str(e)})

        raise