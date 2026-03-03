# Speech-to-Text tool
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Optional, TypedDict, Iterable

import torch
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment

from .audio_preprocess import preprocess_audio

# Whisper does not work well with Apple MPS GPUs
device = "cuda" if torch.cuda.is_available() else "cpu"

class STTError(RuntimeError):
    pass

class STTMeta(TypedDict):
    language: Optional[str]
    duration_s: Optional[float]
    model_size: str
    task: str

# Stores individual segment information
@dataclass(frozen=True)
class STTSegment:
    start_s: float 
    end_s: float 
    text: str

# Stores global information of transcript
@dataclass(frozen=True)
class STTResult:
    language: Optional[str]
    duration_s: Optional[float]
    text: str
    segments: list[STTSegment]

def _seconds_to_srt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)

    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def save_stt_result(
        result: STTResult,
        output_path: str | Path,
        formats: list[Literal["json", "srt"]]=["json", "srt"]
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True) # makes a directory for saving transcripts

    for fmt in formats:
        path = output_path.with_suffix(f".{fmt}")

        if fmt == "json":
            payload = {
                "language": result.language,
                "text": result.text,
                "segments": [
                    {"start_s": s.start_s, "end_s": s.end_s, "text": s.text}
                    for s in result.segments
                ]
            }
            with open(path, "w") as f:
                json.dump(payload, f)

        elif fmt == "srt":
            with open(path, "w") as f:
                for idx, segment in enumerate(result.segments):
                    f.write(f"{idx}\n")
                    f.write(f"{_seconds_to_srt_time(segment.start_s)} --> {_seconds_to_srt_time(segment.end_s)}\n")
                    f.write(f"{segment.text}\n\n")

def transcription_core(
        input_audio_path: str | Path,
        *,
        work_dir: str="tmp/stt",
        model_size: Literal["base", "tiny", "small", "medium"]="small",
        # Decode kwargs
        beam_size: int=5,
        temperature: float=0.0,
        language: Optional[str]=None,
        task: Literal["transcribe", "translate"]="transcribe"
) -> tuple[Iterable[Segment], STTMeta]:
    """ 
    This is the core transcription function using Faster-Whisper.

    Args:
        input_audio_path (str | Path) - path to the input audio file
        work_dir (str) - temporary directory to store preprocessed audio file(s); defaults to "tmp/stt"
        model_size (Literal["base", "tiny", "small", "medium"]) - a selection of available Whisper models; defaults to "small"
        beam_size (int) - the number of candidate transcriptions Whisper considers simultaneously; defaults to 5
        temperature (float) - controls how confident/random the results are; defaults to 0.0 (least randomness)
        language (str, Optional) - the language in the audio file; defaults to None
        task (Literal["transcribe", "translate"]) - the kind of task to be performed; defaults to "transcribe"

    Returns:
        tuple[Iterator, STTMeta] - returns a tuple of transcribed segment iterator and metadata
    """
    input_audio_path = Path(input_audio_path)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True) # temporary directory to store preprocessed audio files

    preprocessed_path = work_dir / f"{input_audio_path.stem}_preprocessed.wav"
    _ = preprocess_audio(input_audio_path, preprocessed_path)

    # Using the `faster-whisper` model. It is faster generates results by streaming.
    try:
        model = WhisperModel(model_size_or_path=model_size, device=device, compute_type="auto")
    except Exception as e:
        raise STTError(f"Failed to load WhisperModel({model_size}): {e}") from e 
    
    # Perform transcription
    try:
        segments, info = model.transcribe(
                            audio=str(preprocessed_path),
                            temperature=temperature,
                            beam_size=beam_size,
                            language=language,
                            task=task
                        )

        if language is None:
            language = info.language

    except Exception as e:
        raise STTError(f"Transcription failed: {e}") from e

    # Store audio metadata
    meta: STTMeta = {
        "language": (language or info.language),
        "duration_s": float(info.duration) if getattr(info, "duration", None) else None,
        "model_size": model_size,
        "task": task
    }

    return segments, meta

def transcription_stream(input_audio_path: str | Path, emit_meta_first: bool=True):
    """
    This function uses the iterator from `transcription_core` for streaming.
    """
    # Get transcription generator and metadata
    segments, meta = transcription_core(input_audio_path)

    if emit_meta_first:
        yield {"type": "meta", **meta}

    parts: list[str] = []
    for segment in segments:
        text = segment.text
        parts.append(text)

        yield {
            "type": "segment",
            "start_s": segment.start,
            "end_s": segment.end,
            "text": text
        }

    # Join all texts
    full_text = " ".join(parts).strip()

    yield {
        "type": "done",
        "text": full_text,
        "language": meta.get("language"),
        "duration_s": meta.get("duration_s")
    }