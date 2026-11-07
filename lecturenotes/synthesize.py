"""Stage 6: turn each aligned section into clean slide note(s), and scan the
whole transcript for administrative mentions.

Per section we ask Claude to merge the on-screen content (printed + handwritten
+ diagram) with what the lecturer said, producing one or more structured
slides. If the content is too much for one slide, the model splits it and
marks the overflow slides as continuations (FR9). Printed slides keep their
image embedded; the generated bullets add the spoken discussion and annotations
around it (FR7/FR8).
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from .config import Config
from .models import AdminItem, FrameKind, Section, SlideNote, TranscriptSegment

_SYS_SECTION = (
    "You convert a lecture segment into clean, comprehensive study-note slides. "
    "You combine what was SHOWN (slide/board text, diagrams, handwritten "
    "annotations) with what was SAID about it. You preserve every substantive "
    "detail, write clearly, and never invent facts. Content hard to read is "
    "marked with '[unclear]'."
)

_SECTION_TEMPLATE = """Here is one segment of a lecture.

ON-SCREEN TYPE: {kind}

AUTHORITATIVE SLIDE/NOTE TEXT (from the lecturer's own uploaded materials; trust this over OCR when present):
{source}

PRINTED/SLIDE TEXT (read from the video frame):
{printed}

HANDWRITTEN / BOARD TEXT:
{handwriting}

DIAGRAM DESCRIPTION:
{diagram}

LECTURER'S ANNOTATIONS (additions on top of the slide/board):
{annotations}

WHAT THE LECTURER SAID DURING THIS SEGMENT:
{spoken}

Produce study-note slides as JSON ONLY:
{{
  "slides": [
    {{"title": "short title", "bullets": ["point", "..."], "note": "optional extra block e.g. diagram explanation"}}
  ]
}}

Rules:
- Merge shown + said content. Attribute board/annotation content and spoken
  elaboration together so nothing is lost.
- Keep bullets self-contained and detailed enough to study from without the video.
- If there is too much for one readable slide (more than ~7 bullets), split into
  multiple slide objects; keep each focused. The renderer marks later ones "(cont.)".
- Do NOT include administrative items (assignments, exams, attendance, deadlines)
  here — those are collected separately.
- If this segment has no instructional content, return {{"slides": []}}.
- Output JSON only."""

_SYS_ADMIN = (
    "You extract administrative announcements from a lecture transcript: "
    "assignments, submission details, exam/quiz info, attendance, deadlines, "
    "and similar logistics. You ignore subject-matter content."
)

_ADMIN_TEMPLATE = """From this lecture transcript, extract ONLY administrative/logistical
announcements (assignments, submissions, exams, quizzes, attendance, deadlines,
grading, office hours, etc.). Ignore all subject content.

Return JSON ONLY:
{{"items": [{{"category": "assignment|exam|attendance|submission|deadline|other", "text": "what was announced, with any date/detail"}}]}}

If there are none, return {{"items": []}}.

TRANSCRIPT:
{transcript}"""


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1].lstrip("json").strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start:end + 1]
    return json.loads(raw)


def _fmt_annotations(section: Section) -> str:
    if not section.extraction.annotations:
        return "(none)"
    out = []
    for a in section.extraction.annotations:
        mark = " [unclear]" if a.uncertain else ""
        out.append(f"- re: {a.refers_to}: {a.text}{mark}")
    return "\n".join(out)


def _synthesize_section(client, section: Section, cfg: Config) -> list[SlideNote]:
    ex = section.extraction
    if ex.kind == FrameKind.OTHER and not section.spoken_text.strip():
        return []
    prompt = _SECTION_TEMPLATE.format(
        kind=ex.kind.value,
        source=section.source_text or "(no uploaded materials matched)",
        printed=ex.printed_text or "(none)",
        handwriting=ex.handwriting_text or "(none)",
        diagram=ex.diagram_description or "(none)",
        annotations=_fmt_annotations(section),
        spoken=section.spoken_text or "(nothing said)",
    )
    resp = client.messages.create(
        model=cfg.workhorse_model,
        max_tokens=2000,
        system=[{"type": "text", "text": _SYS_SECTION, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}],
    )
    data = _parse_json("".join(b.text for b in resp.content if b.type == "text"))

    # prefer the crisp source-slide image when we have a confident match;
    # otherwise embed the video keyframe for visual content.
    if section.source_image_path:
        embed = section.source_image_path
    elif ex.kind in (FrameKind.SLIDE, FrameKind.ANNOTATED_SLIDE, FrameKind.WHITEBOARD):
        embed = section.keyframe.image_path
    else:
        embed = None

    slides: list[SlideNote] = []
    for i, s in enumerate(data.get("slides", []) or []):
        slides.append(SlideNote(
            title=str(s.get("title", "Untitled")).strip(),
            bullets=[str(b).strip() for b in s.get("bullets", []) if str(b).strip()],
            continued=(i > 0),
            image_path=embed if i == 0 else None,  # embed image on the first slide only
            note=str(s.get("note", "")).strip(),
        ))
    return slides


def synthesize_sections(
    sections: list[Section],
    cfg: Config,
    progress: Optional[Callable[[float], None]] = None,
) -> list[SlideNote]:
    from anthropic import Anthropic  # type: ignore
    client = Anthropic(api_key=cfg.require_api_key())

    ordered: dict[int, list[SlideNote]] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=cfg.max_api_concurrency) as pool:
        futs = {pool.submit(_synthesize_section, client, sec, cfg): i
                for i, sec in enumerate(sections)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                ordered[i] = fut.result()
            except Exception:
                ordered[i] = []
            done += 1
            if progress:
                progress(done / max(len(sections), 1))

    slides: list[SlideNote] = []
    for i in range(len(sections)):
        slides.extend(ordered.get(i, []))
    return slides


def scan_admin(
    transcript: list[TranscriptSegment],
    cfg: Config,
) -> list[AdminItem]:
    """Single pass over the full transcript for logistics (FR10)."""
    from anthropic import Anthropic  # type: ignore
    client = Anthropic(api_key=cfg.require_api_key())

    text = "\n".join(f"[{int(s.start)}s] {s.text}" for s in transcript)
    resp = client.messages.create(
        model=cfg.workhorse_model,
        max_tokens=1500,
        system=[{"type": "text", "text": _SYS_ADMIN, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": _ADMIN_TEMPLATE.format(transcript=text)}],
    )
    try:
        data = _parse_json("".join(b.text for b in resp.content if b.type == "text"))
    except Exception:
        return []
    return [
        AdminItem(category=str(it.get("category", "other")),
                  text=str(it.get("text", "")).strip(), timestamp=0.0)
        for it in data.get("items", []) or [] if str(it.get("text", "")).strip()
    ]
