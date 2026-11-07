"""Orchestrates the full pipeline and reports progress per stage.

A progress callback receives (stage_name, fraction_0_to_1). Transcription and
frame detection are independent and run in parallel to save wall-clock time.
"""
from __future__ import annotations

import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

from . import align, build_pptx, frames, ingest, synthesize, transcribe, vision
from .config import Config
from .models import LectureResult

ProgressCb = Callable[[str, float], None]


def _noop(stage: str, frac: float) -> None:
    pass


def run_pipeline(
    video_path: str | Path,
    output_pptx: str | Path,
    cfg: Config,
    progress: Optional[ProgressCb] = None,
) -> LectureResult:
    progress = progress or _noop
    video_path = Path(video_path)
    output_pptx = Path(output_pptx)
    title = video_path.stem.replace("_", " ").replace("-", " ").title()

    work = Path(cfg.workdir) if cfg.workdir else Path(
        tempfile.mkdtemp(prefix="lecturenotes_", dir=output_pptx.parent))
    work.mkdir(parents=True, exist_ok=True)

    # --- stage 1: ingest ---
    progress("ingest", 0.0)
    audio = ingest.extract_audio(video_path, work / "audio.wav")
    progress("ingest", 1.0)

    # --- stages 2 & 3 in parallel: transcribe (GPU) + frame detection (CPU) ---
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_tx = pool.submit(
            transcribe.transcribe, audio, cfg,
            lambda fr: progress("transcribe", fr))
        f_kf = pool.submit(
            frames.detect_keyframes, video_path, work, cfg,
            lambda fr: progress("frames", fr))
        transcript = f_tx.result()
        keyframes = f_kf.result()

    # --- stage 4: vision extraction ---
    progress("vision", 0.0)
    extractions = vision.extract_all(
        keyframes, cfg, lambda fr: progress("vision", fr))

    # --- stage 5: align ---
    progress("align", 0.0)
    sections = align.build_sections(keyframes, extractions, transcript)
    progress("align", 1.0)

    # --- stage 6: synthesize + admin scan ---
    slides = synthesize.synthesize_sections(
        sections, cfg, lambda fr: progress("synthesize", fr))
    progress("admin", 0.0)
    admin = synthesize.scan_admin(transcript, cfg)
    progress("admin", 1.0)

    # --- stage 7: build pptx ---
    progress("pptx", 0.0)
    build_pptx.build_pptx(title, slides, admin, output_pptx)
    progress("pptx", 1.0)

    result = LectureResult(title=title, slides=slides, admin=admin, transcript=transcript)

    if cfg.keep_intermediates:
        _dump_intermediates(work, result)
    progress("done", 1.0)
    return result


def _dump_intermediates(work: Path, result: LectureResult) -> None:
    (work / "transcript.txt").write_text(
        "\n".join(f"[{s.start:7.1f}] {s.text}" for s in result.transcript),
        encoding="utf-8")
    (work / "notes.json").write_text(
        json.dumps(
            {"title": result.title,
             "slides": [asdict(s) for s in result.slides],
             "admin": [asdict(a) for a in result.admin]},
            indent=2, ensure_ascii=False),
        encoding="utf-8")
