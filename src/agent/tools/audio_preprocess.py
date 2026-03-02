# Audio preprocessing module

from __future__ import annotations 

from pathlib import Path 

import ffmpeg

class AudioPreprocessError(RuntimeError):
    pass

def check_audio_length(input_path: str | Path) -> float:
    input_path = Path(input_path)
    try:
        probe_result = ffmpeg.probe(input_path)
        duration = float(probe_result["format"]["duration"])
        return duration
    except ffmpeg.Error as e:
        raise AudioPreprocessError(
            f"FFmpeg preprocessing failed: {e.stderr}"
        ) from e

def split_audio(input_path: str | Path, output_path: str | Path, chunk_duraation: float=600.0) -> None:
    input_path = Path(input_path)
    ext = input_path.suffix
    (
        ffmpeg
        .input(input_path)
        .output(
            f"{output_path}_%3d.{ext}",
            f="segment",
            segment_time=chunk_duraation,
            c="copy"
        )
        .overwrite_output()
        .run(quiet=True)
    )

def preprocess_audio(
        input_path: str | Path,
        output_path: str | Path,
        *,
        target_freq: int=16000,
        mono: bool=True,
        loudnorm: bool=True,
        bandlimit: bool=False,
        trim_leading_trailing_silence: bool=False,
        overwrite: bool=True
) -> str | Path:
    """ 
    This function preprocesses audio file for transcription. The file
    is preprocessed, converted to .wav and saved in a tmp/ directory.

    Args:
        (i) input_path (str | Path): path to audio file
        (ii) output_path (str | Path): path to save preprocessed audio file
        (iii) target_freq (int): target frequency in Hz; defaults to 16000 Hz
        (iv) loudnorm (bool): loudness normalization; defaults to True
        (v) bandlimit (bool): applies high and lowpass filters to signal; defaults to False
        (vi) trim_leading_trailing_silence (bool): trims leading trailing silences; defaults to False
        (vii) overwrite (bool): overwrites audio file

    Returns:
        output path (str | Path): path to preprocessed audio file
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        stream = ffmpeg.input(input_path)
        audio = stream.audio

        if loudnorm:
            audio = audio.filter("loudnorm", I=-16, LRA=11, TP=-1.5)

        if bandlimit:
            audio = audio.filter("highpass", f=80)
            audio = audio.filter("lowpass", f=8000)

        if trim_leading_trailing_silence:
            audio = audio.filter(
                "silenceremove",
                start_periods=1,
                start_silence=1,
                start_threshold="-50dB",
                stop_periods=1,
                stop_silence=1,
                stop_threshold="-50dB"
            )

        stream = ffmpeg.output(audio, str(output_path), ar=target_freq, ac=1 if mono else 2)

        if overwrite:
            stream = stream.overwrite_output()

        stream.run(quiet=False)

    except ffmpeg.Error as e:
        raise AudioPreprocessError(
            f"FFmpeg preprocessing failed: {e.stderr}"
        ) from e 
    
    return output_path