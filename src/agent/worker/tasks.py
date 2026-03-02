from __future__ import annotations

from pathlib import Path
from typing import Any

from .celery_app import celery_app
from ..tools.transcription.service import (
    STTError,
    STTResult,
    perform_transcription
)

def stt_result_to_dict(result: STTResult) -> dict[str, Any]:
    """Converts STTResult to a dictionary."""
    return {
        "language": result.language,
        "duration_s": result.duration_s,
        "text": result.text,
        "segments": [
            {"start_s": segment.start_s, "end_s": segment.end_s, "text": segment.text}
            for segment in result.segments
        ]
    }

@celery_app.task(bind=True, name="task.transcribe_audio")
def transcribe_audio(self, file_path: str) -> dict[str, Any]:
    """Celery task for audio transcription."""
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Missing audio file: {file_path}")

        results: STTResult = perform_transcription(file_path)
        return stt_result_to_dict(results)

    except STTError as e:
        raise e

    except Exception as e:
        raise RuntimeError(f"Transcription error: {e}") from e