"""
Celery tasks for audio transcription and document ingestion.
"""

from __future__ import annotations

import json

from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from .celery_app import celery_app
from agent.memory.redis_config import redis_jobs_sync as r
from agent.ingestion.pipeline import load_and_split
from agent.tools.transcription.stt import transcription_stream
from agent.vector_stores.factory import create_qdrant_vector_store

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "storage" / "transcripts"
STORAGE_DIR = REPO_ROOT / "storage" / "ingestion"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

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
            raise RuntimeError("Transcription stream ended without a `done` event.")

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

@celery_app.task(bind=True, name="task.ingest_documents")
def ingest_documents(
    self,
    ingestion_id: str,
    session_id: str,
    user_id: str | None,
    file_paths: Sequence[Path | str],
    initial_results: dict[str, Any]
) -> dict[str, Any]:
    """
    Celery task for RAG document ingestion. It loads and splits documents, adds
    metadata and stores them in a chosen vector database. The task status and results
    are stored in Redis for polling.

    Args:
        ingestion_id (str): Unique ID for this ingestion job
        session_id (str): Session ID to associate with this ingestion
        user_id (str | None): Optional user ID for tracking
        file_paths (Sequence[Path | str]): List of document file paths to ingest
        initial_results (dict[str, Any]): Initial results to include in the final output

    Returns:
        dict[str, Any]: A dictionary containing ingestion results and metadata
    """
    try:
        r.set(f"ingesion:{ingestion_id}:status", "running")
        r.delete(f"ingestion:{ingestion_id}:error")

        paths = [Path(fp) for fp in file_paths]
        chunks = load_and_split(paths)

        for chunk in chunks:
            chunk.metadata.update(
                {
                    "ingestion_id": ingestion_id,
                    "session_id": session_id,
                    "user_id": user_id
                }
            )

        ids = [uuid4().hex for _ in chunks]
        vector_store = create_qdrant_vector_store()
        vector_store.add_documents(chunks, ids=ids)

        result = {
            "ingestion_id": ingestion_id,
            "session_id": session_id,
            "user_id": user_id,
            "file_count": len(file_paths),
            "chunk_count": len(chunks),
            "document_ids": ids,
            "results": initial_results
        }

        r.set(f"ingestion:{ingestion_id}:result", json.dumps(result))
        r.set(f"ingestion:{ingestion_id}:status", "done")
        r.set(f"session_id:{session_id}:ingestion:{ingestion_id}", json.dumps(result))

        return result
    except Exception as e:
        r.set(f"ingestion:{ingestion_id}:status", "error")
        r.set(f"ingestion:{ingestion_id}:error", str(e))
        raise
