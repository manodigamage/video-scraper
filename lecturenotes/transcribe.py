"""Stage 2: transcribe audio locally with faster-whisper (GPU)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

from .config import Config
from .models import TranscriptSegment


def _load_model(model_name: str, device: str, compute: str, attempts: int = 3):
    """Load (first run: download) the Whisper model, retrying transient
    network/socket failures that can interrupt the Hugging Face download."""
    from faster_whisper import WhisperModel  # type: ignore
    last = None
    for i in range(attempts):
        try:
            return WhisperModel(model_name, device=device, compute_type=compute)
        except Exception as e:  # noqa: BLE001 - surface a clear message after retries
            last = e
            time.sleep(3 * (i + 1))
    raise RuntimeError(
        f"Could not load the Whisper model '{model_name}' after {attempts} tries. "
        f"The first run downloads it (~1-3 GB) from Hugging Face, so this is usually "
        f"a network/VPN/firewall issue. Last error: {last}")


def _pick_model_by_vram(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if gb >= 10:
                return "large-v3"
            if gb >= 6:
                return "distil-large-v3"
            return "medium"
    except Exception:
        pass
    # CPU / unknown: smaller, still decent
    return "small"


def transcribe(
    audio_path: str | Path,
    cfg: Config,
    progress: Optional[Callable[[float], None]] = None,
) -> list[TranscriptSegment]:
    """Return full timestamped transcript. Runs on GPU if available."""
    model_name = _pick_model_by_vram(cfg.whisper_model)

    device, compute = "cuda", cfg.compute_type
    try:
        import torch  # type: ignore
        if not torch.cuda.is_available():
            device, compute = "cpu", "int8"
    except Exception:
        device, compute = "cpu", "int8"

    model = _load_model(model_name, device, compute)

    segments_iter, info = model.transcribe(
        str(audio_path),
        language=cfg.whisper_language,
        vad_filter=True,                 # skip long silences -> faster, cleaner
        beam_size=5,
    )
    total = float(getattr(info, "duration", 0.0)) or 1.0

    out: list[TranscriptSegment] = []
    for seg in segments_iter:
        out.append(TranscriptSegment(start=seg.start, end=seg.end, text=seg.text.strip()))
        if progress:
            progress(min(seg.end / total, 1.0))
    if progress:
        progress(1.0)
    return out


def transcript_to_text(segments: list[TranscriptSegment]) -> str:
    return "\n".join(f"[{s.start:7.1f}] {s.text}" for s in segments)
