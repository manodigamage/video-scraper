"""Integration smoke test for the LOCAL stages of the pipeline.

Generates a short synthetic lecture video (3 slides, one annotated) plus a PDF
of "lecturer materials", then runs the full pipeline with the GPU transcription
and the paid Claude API calls STUBBED OUT. This exercises ingest, frame
sampling + perceptual-hash dedup, transcript alignment, materials matching, and
PPTX rendering on real files — no GPU and no API spend.

Run:  python tests/smoke_test.py
Needs: ffmpeg on PATH + the pip deps in requirements.txt (anthropic/faster-whisper not required).
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lecturenotes import pipeline, synthesize, transcribe, vision
from lecturenotes.config import Config
from lecturenotes.models import (AdminItem, FrameExtraction, FrameKind,
                                 SlideNote, TranscriptSegment)

# schedule: (text, start, end) for what is on screen when
SLIDES = [
    ("Lecture 1: Neural Networks\n- perceptron\n- activation functions", 0, 6),
    ("Backpropagation\n- gradient descent\n- chain rule\n- learning rate", 6, 12),
    ("Backpropagation\n- gradient descent\n- chain rule\n- learning rate\n[HANDWRITTEN] note: watch vanishing gradients!", 12, 18),
    ("Convolutional Networks\n- filters\n- pooling\n- stride and padding", 18, 24),
]


def _font(size: int):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def make_slide_png(text: str, path: Path, annotated: bool):
    img = Image.new("RGB", (1280, 720), "white")
    d = ImageDraw.Draw(img)
    d.multiline_text((80, 80), text, fill="black", font=_font(44), spacing=18)
    if annotated:
        # a prominent "handwritten" red scrawl so it registers as a new state
        d.line([(900, 300), (1150, 320), (950, 420), (1180, 460)], fill="red", width=8)
        d.text((880, 500), "!!!", fill="red", font=_font(80))
    img.save(path)


def build_video(work: Path) -> Path:
    frames_dir = work / "slides"
    frames_dir.mkdir(parents=True, exist_ok=True)
    listing = []
    for i, (text, start, end) in enumerate(SLIDES):
        png = frames_dir / f"s{i}.png"
        make_slide_png(text, png, annotated=i == 2)
        listing.append(f"file '{png}'")
        listing.append(f"duration {end - start}")
    listing.append(f"file '{frames_dir / 's3.png'}'")  # concat demuxer needs last repeated
    (work / "list.txt").write_text("\n".join(listing), encoding="utf-8")

    out = work / "lecture.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(work / "list.txt"),
         "-f", "lavfi", "-t", "24", "-i", "anullsrc=r=16000:cl=mono",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "5",
         "-c:a", "aac", "-shortest", str(out)],
        capture_output=True, text=True, check=True)
    return out


def build_pdf(work: Path) -> Path:
    import pymupdf
    pdf = work / "materials.pdf"
    doc = pymupdf.open()
    for text, *_ in SLIDES[:1] + SLIDES[1:2] + SLIDES[3:4]:  # slide1, slide2, slide3
        page = doc.new_page(width=1280, height=720)
        page.insert_text((72, 100), text, fontsize=28)
    doc.save(str(pdf))
    doc.close()
    return pdf


def _text_for_timestamp(ts: float) -> tuple[str, FrameKind]:
    for text, start, end in SLIDES:
        if start <= ts < end:
            kind = FrameKind.ANNOTATED_SLIDE if "[HANDWRITTEN]" in text else FrameKind.SLIDE
            return text.replace("[HANDWRITTEN] ", ""), kind
    return SLIDES[-1][0], FrameKind.SLIDE


# ---- stubs replacing GPU transcription and paid API ----

def fake_transcribe(audio_path, cfg, progress=None):
    segs = [
        TranscriptSegment(1, 5, "Welcome, today we cover neural networks and the perceptron."),
        TranscriptSegment(7, 11, "Backpropagation uses gradient descent and the chain rule."),
        TranscriptSegment(13, 17, "Be careful about vanishing gradients with deep networks."),
        TranscriptSegment(19, 23, "Convolutional networks use filters, pooling, stride and padding. "
                                  "Assignment one is due this Friday on the portal."),
    ]
    if progress:
        progress(1.0)
    return segs


def fake_extract_all(keyframes, cfg, progress=None):
    out = {}
    for kf in keyframes:
        text, kind = _text_for_timestamp(kf.timestamp)
        hw = "watch vanishing gradients!" if kind == FrameKind.ANNOTATED_SLIDE else ""
        out[kf.index] = FrameExtraction(
            frame_index=kf.index, kind=kind, printed_text=text,
            handwriting_text=hw, model_used="stub")
    if progress:
        progress(1.0)
    return out


def fake_synthesize_sections(sections, cfg, progress=None):
    slides = []
    for sec in sections:
        base = sec.source_text or sec.extraction.printed_text
        title = base.splitlines()[0][:60] if base else "Section"
        bullets = [ln.strip("- ").strip() for ln in base.splitlines()[1:] if ln.strip()]
        if sec.spoken_text:
            bullets.append("Discussion: " + sec.spoken_text)
        slides.append(SlideNote(
            title=title, bullets=bullets or ["(no content)"],
            image_path=sec.source_image_path or sec.keyframe.image_path))
    if progress:
        progress(1.0)
    return slides


def fake_scan_admin(transcript, cfg):
    return [AdminItem(category="assignment", text="Assignment 1 due Friday on the portal.", timestamp=0.0)]


def run(with_materials: bool) -> None:
    work = Path(tempfile.mkdtemp(prefix="smoke_"))
    video = build_video(work)
    materials = build_pdf(work) if with_materials else None

    # patch out GPU + API
    transcribe.transcribe = fake_transcribe
    vision.extract_all = fake_extract_all
    synthesize.synthesize_sections = fake_synthesize_sections
    synthesize.scan_admin = fake_scan_admin

    cfg = Config(api_key="stub", keep_intermediates=True, workdir=str(work / "wd"))
    out_pptx = work / "out.pptx"

    seen = []
    result = pipeline.run_pipeline(
        video, out_pptx, cfg, materials_path=materials,
        progress=lambda s, f: seen.append((s, f)))

    assert out_pptx.exists(), "pptx not written"
    from pptx import Presentation
    prs = Presentation(str(out_pptx))
    n = len(prs.slides)

    tag = "WITH materials" if with_materials else "NO materials"
    print(f"[{tag}] keyframes->sections slides={len(result.slides)} admin={len(result.admin)} "
          f"pptx_slides={n}")
    assert len(result.slides) >= 3, "expected >=3 content slides"
    assert len(result.admin) == 1
    assert n >= 5  # title + >=3 content + admin
    stages = {s for s, _ in seen}
    assert {"ingest", "transcribe", "frames", "vision", "align", "synthesize", "pptx"} <= stages
    if with_materials:
        assert "materials" in stages
        matched = [s for s in result.slides if s.image_path and "materials" in str(s.image_path)]
        print(f"           source-image slides matched: {len(matched)}")
    print(f"           output: {out_pptx}")


if __name__ == "__main__":
    run(with_materials=False)
    run(with_materials=True)
    print("\nSMOKE TEST PASSED")
