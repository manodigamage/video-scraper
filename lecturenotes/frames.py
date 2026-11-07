"""Stage 3: content-aware visual-state detection.

A single pixel-similarity threshold can't separate "lecturer drew on the slide"
(ignore) from "a near-identical new slide" (keep). We combine two signals:

  * PIXEL / TEMPORAL change — catches big abrupt changes (a visually different
    slide, an image) and, being a per-step delta, treats gradual hand-drawn
    annotation as "no boundary".
  * OCR CONTENT fingerprint — the reliable discriminator for template-twin
    slides (same border/header, body differs). We OCR on a bounded cadence,
    crop away the app/browser chrome, drop noise tokens, and start a new state
    when the printed-text fingerprint diverges from the previous one — even when
    the pixels barely moved. Hand-drawn ink mostly doesn't survive as clean
    printed tokens, so annotation keeps the fingerprint stable.

Per state we keep the last stable frame (the most-complete / most-annotated).
"""
from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Callable, Optional

import cv2

from .config import Config
from .models import Keyframe

_SMALL = (320, 180)
_PIXEL_DELTA = 25
# app/browser chrome and boilerplate words to ignore in the text fingerprint
_STOP = {"menu", "sign", "pdf", "create", "support", "vector", "machin", "find",
         "text", "tools", "convert", "share", "all", "read", "edit", "ask",
         "assistant", "university", "texas", "austin", "group", "machine",
         "learning", "example"}

_OCR = None  # lazily created RapidOCR singleton


def _ocr_engine():
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore
        _OCR = RapidOCR()
    return _OCR


def _sample_frames(video_path, out_dir: Path, fps: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "frame_%06d.jpg")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path),
         "-vf", f"fps={fps},scale='min(1280,iw)':-2", "-q:v", "4", pattern],
        capture_output=True, text=True, check=True)
    frames = sorted(out_dir.glob("frame_*.jpg"))
    return [(i / fps, p) for i, p in enumerate(frames)]


def _small_gray(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return cv2.GaussianBlur(cv2.resize(img, _SMALL), (3, 3), 0)


def _change_ratio(a, b) -> float:
    return float((cv2.absdiff(a, b) > _PIXEL_DELTA).mean())


def _ocr_fingerprint(path: Path) -> set:
    """Printed-text token set from the content area (chrome cropped, noise and
    boilerplate dropped)."""
    img = cv2.imread(str(path))
    if img is None:
        return set()
    h, w = img.shape[:2]
    crop = img[int(h * 0.12):int(h * 0.94), int(w * 0.02):int(w * 0.96)]
    res, _ = _ocr_engine()(crop)
    text = " ".join(r[1] for r in (res or []))
    out = set()
    for t in text.replace("(", " ").replace(")", " ").split():
        tl = t.lower()
        if len(tl) >= 3 and any(c.isalpha() for c in tl) and tl not in _STOP:
            out.add(tl)
    return out


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / max(len(a | b), 1)


def detect_keyframes(
    video_path,
    out_dir,
    cfg: Config,
    progress: Optional[Callable[[float], None]] = None,
) -> list[Keyframe]:
    out_dir = Path(out_dir)
    fps = cfg.sample_fps
    sampled = _sample_frames(video_path, out_dir / "_sampled", fps)
    if not sampled:
        return []

    # OCR cadence, but capped so a long video never OCRs more than max_ocr_samples
    ocr_step = max(1, round(cfg.ocr_interval_s * fps),
                   math.ceil(len(sampled) / max(cfg.max_ocr_samples, 1)))
    min_frames = max(1, int(cfg.min_state_seconds * fps))

    # --- pass 1: per-frame pixel delta; OCR fingerprint on a cadence ---
    dpix = [0.0] * len(sampled)
    fps_at = {}
    prev_small = None
    for i, (ts, path) in enumerate(sampled):
        small = _small_gray(path)
        dpix[i] = 1.0 if prev_small is None else (
            _change_ratio(prev_small, small) if small is not None else 0.0)
        prev_small = small if small is not None else prev_small
        if i % ocr_step == 0:
            fps_at[i] = _ocr_fingerprint(path)
        if progress:
            progress((i + 1) / len(sampled) * 0.9)

    # --- pass 2: boundaries = abrupt pixel jumps + OCR content changes ---
    # A content change is detected late (at the next OCR sample); refine it to
    # the actual turn by taking the biggest pixel jump since the last OCR sample.
    boundaries = {0}
    for i, d in enumerate(dpix):
        if d > cfg.change_threshold:
            boundaries.add(i)
    ocr_idx = sorted(fps_at)
    for a, b in zip(ocr_idx, ocr_idx[1:]):
        if _jaccard(fps_at[b], fps_at[a]) < cfg.split_jaccard:
            k = max(range(a + 1, b + 1), key=lambda x: dpix[x])
            boundaries.add(k)

    # --- segment between boundaries; representative = last frame of each state ---
    bs = sorted(boundaries)
    kept = out_dir / "keyframes"
    kept.mkdir(parents=True, exist_ok=True)
    keyframes: list[Keyframe] = []
    for start, end in zip(bs, bs[1:] + [len(sampled)]):
        if end - start < min_frames:
            continue
        idx = len(keyframes)
        rep_path = sampled[end - 1][1]            # last frame before next turn
        dst = kept / f"keyframe_{idx:03d}.jpg"
        dst.write_bytes(Path(rep_path).read_bytes())
        keyframes.append(Keyframe(index=idx, timestamp=sampled[start][0], image_path=str(dst)))

    for i, kf in enumerate(keyframes):
        kf.end_timestamp = keyframes[i + 1].timestamp if i + 1 < len(keyframes) else None
    if progress:
        progress(1.0)
    return keyframes
