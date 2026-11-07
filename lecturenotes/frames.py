"""Stage 3: find distinct visual states (keyframes) cheaply and locally.

Strategy: sample the video at a low fps, compute a perceptual hash of each
sampled frame, and collapse runs of near-identical frames into a single
"visual state". For each state we keep the LAST frame of the run, because on
slides/boards content is usually most complete just before it changes
(progressive annotations). This keeps the number of frames sent to the vision
model small (dozens, not thousands) — the main cost/speed lever.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Optional

from .config import Config
from .models import Keyframe


def _sample_frames(video_path: str | Path, out_dir: Path, fps: float) -> list[tuple[float, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "frame_%06d.jpg")
    # scale so the longest edge <= 1280 for cheap hashing; q=4 is plenty here
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path),
         "-vf", f"fps={fps},scale='min(1280,iw)':-2",
         "-q:v", "4", pattern],
        capture_output=True, text=True, check=True,
    )
    frames = sorted(out_dir.glob("frame_*.jpg"))
    return [(i / fps, p) for i, p in enumerate(frames)]


def _phash(path: Path) -> "object":
    import imagehash  # type: ignore
    from PIL import Image  # type: ignore
    with Image.open(path) as im:
        return imagehash.phash(im)


def detect_keyframes(
    video_path: str | Path,
    out_dir: str | Path,
    cfg: Config,
    progress: Optional[Callable[[float], None]] = None,
) -> list[Keyframe]:
    out_dir = Path(out_dir)
    sampled = _sample_frames(video_path, out_dir / "_sampled", cfg.sample_fps)
    if not sampled:
        return []

    min_run = max(1, int(cfg.min_state_seconds * cfg.sample_fps))

    # Build runs of visually-similar consecutive frames.
    runs: list[list[tuple[float, Path, object]]] = []
    prev_hash = None
    for i, (ts, path) in enumerate(sampled):
        h = _phash(path)
        if prev_hash is not None and (h - prev_hash) <= cfg.phash_distance_threshold:
            runs[-1].append((ts, path, h))
        else:
            runs.append([(ts, path, h)])
        prev_hash = h
        if progress:
            progress((i + 1) / len(sampled))

    keyframes: list[Keyframe] = []
    kept_dir = out_dir / "keyframes"
    kept_dir.mkdir(parents=True, exist_ok=True)
    idx = 0
    for run in runs:
        if len(run) < min_run:
            continue  # too brief -> transition/flicker
        ts, src, h = run[-1]            # last (most complete) frame of the run
        dst = kept_dir / f"keyframe_{idx:03d}.jpg"
        dst.write_bytes(Path(src).read_bytes())
        kf = Keyframe(index=idx, timestamp=run[0][0], image_path=str(dst), phash=str(h))
        kf.end_timestamp = run[-1][0]
        keyframes.append(kf)
        idx += 1

    # end_timestamp = start of the next kept state (for transcript alignment)
    for i, kf in enumerate(keyframes):
        kf.end_timestamp = keyframes[i + 1].timestamp if i + 1 < len(keyframes) else None
    if progress:
        progress(1.0)
    return keyframes
