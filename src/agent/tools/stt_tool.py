# Speech-to-Text tool

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import whisper 

from .audio_preprocess import preprocess_audio

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
    language_probability: Optional[str]
    duration_s: Optional[float]
    text: str
    segments: list[STTSegment]

def perform_transcription(
        input_audio_path: str | Path,
        *,
        work_dir: str="tmp/stt",
        model_size: Literal["base", "tiny", "small", "medium"]="small",
        device: Literal["cpu", "gpu", "auto"]="auto",
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
        (iv) device (Literal["cpu", "gpu", "auto"]) - type of device that is available; defaults to "auto"
        (v) beam_size (int) - the number of candidate transcriptions Whisper considers simultaneously; defaults to 5
        (vi) temperature (float) - controls how confident/random the results are; defaults to 0.0 (least randomness)
        (v) language (str, Optional) - the language in the audio file; defaults to None
        (vi) task (Literal["transcribe", "translate"]) - the kind of task to be performed; defaults to "transcribe"
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
            str(preprocessed_path),
            temperature=temperature,
            beam_size=beam_size,
            language=language,
            task=task,
            verbose=False
        )

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