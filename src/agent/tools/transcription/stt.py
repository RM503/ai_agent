# Speech-to-Text tool
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch
import whisper 

from .audio_preprocess import preprocess_audio

# Whisper does not work well with Apple MPS GPUs
device = "cuda" if torch.cuda.is_available() else "cpu"

class STTError(RuntimeError):
    pass

@dataclass(frozen=True)
class STTSegment:
    start_s: float 
    end_s: float 
    text: str

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

def perform_transcription(
        input_audio_path: str | Path,
        *,
        work_dir: str="tmp/stt",
        model_size: Literal["base", "tiny", "small", "medium"]="small",
        # Decode kwargs
        beam_size: int=5,
        temperature: float=0.0,
        language: Optional[str]=None,
        task: Literal["transcribe", "translate"]="transcribe"
) -> STTResult:
    """ 
    This function performs transcription/translation using OpenAI Whisper model (locally).

    Args:
        (i) input_audio_path (str | Path) - path to the input audio file
        (ii) work_dir (str) - temporary directory to store preprocessed audio file(s); defaults to "tmp/stt"
        (iii) model_size (Literal["base", "tiny", "small", "medium"]) - a selection of available Whisper models; defaults to "small"
        (iv) beam_size (int) - the number of candidate transcriptions Whisper considers simultaneously; defaults to 5
        (v) temperature (float) - controls how confident/random the results are; defaults to 0.0 (least randomness)
        (vi) language (str, Optional) - the language in the audio file; defaults to None
        (vii) task (Literal["transcribe", "translate"]) - the kind of task to be performed; defaults to "transcribe"
    """
    input_audio_path = Path(input_audio_path)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True) # temporary directory to store preprocessed audio files

    preprocessed_path = work_dir / f"{input_audio_path.stem}_preprocessed.wav"
    _ = preprocess_audio(input_audio_path, preprocessed_path)
    
    # Load model
    try:
        model = whisper.load_model(name=model_size, device=device)
    except Exception as e:
        raise STTError(f"Failed to load WhisperModel({model_size}): {e}") from e 
    
    # Perform transcription
    try:
        raw = model.transcribe(
            audio=str(preprocessed_path),
            temperature=temperature,
            beam_size=beam_size,
            language=language,
            task=task,
            verbose=False
        )

        if language is None:
            language = raw["language"]

    except Exception as e:
        raise STTError(f"Transcription failed: {e}") from e
    
    # Store transcribed audio segments
    segments = [
        STTSegment(start_s=segment["start"], end_s=segment["end"], text=segment["text"].strip())
        for segment in raw.get("segments", [])
    ]

    duration_s = segments[-1].end_s if segments else None

    return STTResult(
        language=language,
        duration_s=duration_s,
        text=raw["text"].strip(),
        segments=segments
    )