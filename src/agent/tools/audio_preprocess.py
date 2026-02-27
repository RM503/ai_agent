from __future__ import annotations 

from pathlib import Path 

import ffmpeg

class AudioPreprocessError(RuntimeError):
    pass 

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
) -> Path:
    """ 
    This function preprocesses audio file for transcription.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        stream = ffmpeg.input(str(input_path))
        filters = []

        if loudnorm:
            filters.append("loudnorm=I=-16:LRA=11:TP=-1.5")

        if bandlimit:
            filters.append("highpass=f=80")
            filters.append("lowpass=f=8000")
        
        if trim_leading_trailing_silence:
            filters.append(
                "silenceremove=start_periods=1:start_silence=1:start_threshold=-50dB:"
                "stop_periods=1:stop_silence=1:stop_threshold=-50dB"
            )
        
        if filters:
            stream = stream.filter_(",".join(filters))

        stream = (
            stream.output(
                str(output_path),
                ac=1 if mono else None,
                ar=target_freq,
                acodec="pcm_s161e"
            )
        )

        if overwrite:
            stream = stream.overwrite_output()

        stream.run(quite=True)

    except ffmpeg.Error as e:
        raise AudioPreprocessError(
            f"FFmpeg preprocessing failed:\n{e.stderr.decode() if e.stderr else e}"
        ) from e 
    
    return output_path