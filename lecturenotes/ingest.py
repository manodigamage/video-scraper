"""Stage 1: ingest the video — probe it and extract audio for transcription."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


class FFmpegError(RuntimeError):
    pass


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FFmpegError(f"{' '.join(cmd[:2])} failed:\n{proc.stderr.strip()}")
    return proc


def probe_duration(video_path: str | Path) -> float:
    """Return the video duration in seconds."""
    proc = _run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json", str(video_path),
    ])
    return float(json.loads(proc.stdout)["format"]["duration"])


def extract_audio(video_path: str | Path, out_wav: str | Path) -> Path:
    """Extract mono 16 kHz PCM wav — the format Whisper expects."""
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    _run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(out_wav),
    ])
    return out_wav
