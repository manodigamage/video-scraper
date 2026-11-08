"""Command-line entry point (Phase 1). The desktop app (Phase 3) wraps the
same `run_pipeline`.

Usage:
    python -m lecturenotes.cli LECTURE.mp4 -o notes.pptx
    python -m lecturenotes.cli LECTURE.mp4 --cheap          # Haiku only, no escalation
    python -m lecturenotes.cli --save-key                   # store API key in keyring
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .config import load_config, save_api_key
from .pipeline import run_pipeline

_STAGE_LABEL = {
    "materials": "Loading reference materials",
    "ingest": "Extracting audio",
    "transcribe": "Transcribing (local GPU)",
    "frames": "Scanning for slides/board changes",
    "vision": "Reading on-screen content",
    "align": "Aligning transcript",
    "synthesize": "Writing notes",
    "admin": "Collecting announcements",
    "pptx": "Building PowerPoint",
    "done": "Done",
}


def _progress(stage: str, frac: float) -> None:
    label = _STAGE_LABEL.get(stage, stage)
    bar = "#" * int(frac * 20)
    sys.stdout.write(f"\r{label:36s} [{bar:<20}] {int(frac*100):3d}%")
    sys.stdout.flush()
    if frac >= 1.0:
        sys.stdout.write("\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Turn a lecture recording into PPTX notes.")
    ap.add_argument("video", nargs="?", help="input video file")
    ap.add_argument("-o", "--output", help="output .pptx path")
    ap.add_argument("-c", "--config", help="optional TOML config file")
    ap.add_argument("-m", "--materials",
                    help="optional lecturer slides/notes (.pdf or .pptx) for accuracy")
    ap.add_argument("--api-key", help="Anthropic API key (else env/keyring)")
    ap.add_argument("--cheap", action="store_true",
                    help="Haiku only, disable Sonnet escalation")
    ap.add_argument("--save-key", action="store_true",
                    help="save the given --api-key to the OS keyring and exit")
    args = ap.parse_args(argv)

    if args.save_key:
        if not args.api_key:
            print("Pass --api-key to save.", file=sys.stderr)
            return 2
        ok = save_api_key(args.api_key)
        print("Saved to keyring." if ok else "Keyring unavailable (install `keyring`).")
        return 0 if ok else 1

    if not args.video:
        ap.error("a video file is required")

    cfg = load_config(args.config, api_key=args.api_key)
    if args.cheap:
        cfg.allow_escalation = False

    video = Path(args.video)
    out = Path(args.output) if args.output else video.with_suffix(".pptx")

    print(f"Input : {video}")
    print(f"Output: {out}\n")
    if args.materials:
        print(f"Materials: {args.materials}")
    t0 = time.time()
    result = run_pipeline(video, out, cfg, materials_path=args.materials, progress=_progress)
    dt = time.time() - t0

    print(f"\n{len(result.slides)} slides + {len(result.admin)} admin items "
          f"in {dt/60:.1f} min")
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
