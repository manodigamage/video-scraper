"""Stage 4: read each keyframe with Claude vision -> structured extraction.

Uses the cheap workhorse model by default and escalates a frame to the
stronger model only when the first pass reports it was unclear (and quality
mode is on). Images are downscaled to `image_max_dim` to control token cost.
Calls run concurrently.
"""
from __future__ import annotations

import base64
import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from .config import Config
from .models import Annotation, FrameExtraction, FrameKind, Keyframe

_SYSTEM = (
    "You are an expert at reading lecture visuals: presentation slides, "
    "physical/shared whiteboards, and slides with handwritten annotations on "
    "top. You transcribe EVERYTHING visible precisely, including messy "
    "handwriting, and you never invent content. If something is illegible, "
    "give your best reading and flag it as uncertain rather than guessing "
    "silently."
)

_PROMPT = """Analyse this single frame captured from a lecture video.

Return ONLY a JSON object with exactly these keys:
{
  "kind": "slide" | "annotated_slide" | "whiteboard" | "other",
  "printed_text": "all machine-printed text on the slide, verbatim, preserving structure with newlines; empty if none",
  "handwriting_text": "transcription of all handwritten/board text; empty if none",
  "diagram_description": "concise description of any diagram/drawing/figure and its labels; empty if none",
  "annotations": [
     {"refers_to": "which part of the slide/board this relates to", "text": "what was written/drawn there", "uncertain": true|false}
  ],
  "unclear": true|false   // true if any important content was hard to read
}

Rules:
- "other" means a talking-head shot, logo, blank/transition frame, or anything with no instructional content.
- For an annotated slide, put the slide's printed text in printed_text and the lecturer's additions in annotations (and handwriting_text).
- Do not summarise or paraphrase the on-screen text — transcribe it.
- Output JSON only, no prose, no code fences."""


def _encode_image(path: str, max_dim: int) -> tuple[str, str]:
    from PIL import Image  # type: ignore
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = min(1.0, max_dim / max(w, h))
        if scale < 1.0:
            im = im.resize((int(w * scale), int(h * scale)))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii"), "image/jpeg"


def _parse(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1].lstrip("json").strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start:end + 1]
    return json.loads(raw)


def _to_extraction(frame_index: int, data: dict, model: str) -> FrameExtraction:
    anns = [
        Annotation(
            refers_to=str(a.get("refers_to", "")),
            text=str(a.get("text", "")),
            uncertain=bool(a.get("uncertain", False)),
        )
        for a in data.get("annotations", []) or []
    ]
    try:
        kind = FrameKind(data.get("kind", "other"))
    except ValueError:
        kind = FrameKind.OTHER
    return FrameExtraction(
        frame_index=frame_index,
        kind=kind,
        printed_text=str(data.get("printed_text", "")).strip(),
        handwriting_text=str(data.get("handwriting_text", "")).strip(),
        diagram_description=str(data.get("diagram_description", "")).strip(),
        annotations=anns,
        unclear=bool(data.get("unclear", False)),
        model_used=model,
    )


def _call(client, model: str, b64: str, media: str) -> dict:
    resp = client.messages.create(
        model=model,
        max_tokens=1500,
        system=[{"type": "text", "text": _SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media, "data": b64}},
                {"type": "text", "text": _PROMPT},
            ],
        }],
    )
    return _parse("".join(b.text for b in resp.content if b.type == "text"))


def extract_frame(client, kf: Keyframe, cfg: Config) -> FrameExtraction:
    b64, media = _encode_image(kf.image_path, cfg.image_max_dim)
    data = _call(client, cfg.workhorse_model, b64, media)
    ext = _to_extraction(kf.index, data, cfg.workhorse_model)
    # escalate hard frames to the stronger model (quality mode)
    if cfg.allow_escalation and ext.unclear and ext.kind != FrameKind.OTHER:
        try:
            data2 = _call(client, cfg.escalation_model, b64, media)
            ext = _to_extraction(kf.index, data2, cfg.escalation_model)
        except Exception:
            pass  # keep the first-pass result
    return ext


def extract_all(
    keyframes: list[Keyframe],
    cfg: Config,
    progress: Optional[Callable[[float], None]] = None,
) -> dict[int, FrameExtraction]:
    from anthropic import Anthropic  # type: ignore
    client = Anthropic(api_key=cfg.require_api_key())

    results: dict[int, FrameExtraction] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=cfg.max_api_concurrency) as pool:
        futures = {pool.submit(extract_frame, client, kf, cfg): kf for kf in keyframes}
        for fut in as_completed(futures):
            kf = futures[fut]
            try:
                results[kf.index] = fut.result()
            except Exception as e:
                results[kf.index] = FrameExtraction(
                    frame_index=kf.index, kind=FrameKind.OTHER,
                    printed_text="", unclear=True, model_used=f"error:{e}",
                )
            done += 1
            if progress:
                progress(done / max(len(keyframes), 1))
    return results
