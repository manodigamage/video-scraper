"""Stage 7: render slide notes + admin appendix into a .pptx (FR11)."""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

from .models import AdminItem, SlideNote

_SLIDE_W = Inches(13.333)
_SLIDE_H = Inches(7.5)
_ACCENT = RGBColor(0x2F, 0x54, 0x96)


def _blank(prs: Presentation):
    return prs.slides.add_slide(prs.slide_layouts[6])


def _add_title(slide, text: str, continued: bool):
    box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.3), Inches(0.9))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = (text + "  (cont.)") if continued else text
    run.font.size = Pt(28)
    run.font.bold = True
    run.font.color.rgb = _ACCENT


def _add_bullets(slide, bullets: list[str], left, top, width, height):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    first = True
    for b in bullets:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        run = p.add_run()
        run.text = "• " + b
        run.font.size = Pt(16)
        p.space_after = Pt(6)


def _title_slide(prs: Presentation, title: str):
    slide = _blank(prs)
    box = slide.shapes.add_textbox(Inches(0.8), Inches(2.8), Inches(11.7), Inches(2))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = title
    run.font.size = Pt(40)
    run.font.bold = True
    run.font.color.rgb = _ACCENT
    sub = tf.add_paragraph()
    sub.alignment = PP_ALIGN.CENTER
    r2 = sub.add_run()
    r2.text = "Auto-generated lecture notes"
    r2.font.size = Pt(18)
    r2.font.color.rgb = RGBColor(0x66, 0x66, 0x66)


def _content_slide(prs: Presentation, note: SlideNote):
    slide = _blank(prs)
    _add_title(slide, note.title, note.continued)

    has_image = bool(note.image_path) and Path(note.image_path).exists()
    if has_image:
        # image on the left, notes on the right
        try:
            slide.shapes.add_picture(
                note.image_path, Inches(0.5), Inches(1.4),
                width=Inches(6.6))
        except Exception:
            has_image = False

    if has_image:
        _add_bullets(slide, note.bullets, Inches(7.4), Inches(1.4),
                     Inches(5.4), Inches(5.6))
        if note.note:
            _add_bullets(slide, [note.note], Inches(0.5), Inches(6.7),
                         Inches(6.6), Inches(0.6))
    else:
        _add_bullets(slide, note.bullets, Inches(0.6), Inches(1.4),
                     Inches(12.1), Inches(5.0))
        if note.note:
            _add_bullets(slide, [note.note], Inches(0.6), Inches(6.2),
                         Inches(12.1), Inches(1.0))


def _admin_slides(prs: Presentation, admin: list[AdminItem]):
    if not admin:
        return
    # group by category, chunk to avoid overfull slides
    by_cat: dict[str, list[str]] = {}
    for it in admin:
        by_cat.setdefault(it.category, []).append(it.text)

    lines: list[str] = []
    for cat, items in by_cat.items():
        lines.append(f"[{cat.upper()}]")
        lines.extend(items)

    chunk = 10
    first = True
    for i in range(0, len(lines), chunk):
        slide = _blank(prs)
        _add_title(slide, "Announcements & Admin", continued=not first)
        first = False
        _add_bullets(slide, lines[i:i + chunk], Inches(0.6), Inches(1.4),
                     Inches(12.1), Inches(5.4))


def build_pptx(title: str, slides: list[SlideNote], admin: list[AdminItem],
               out_path: str | Path) -> Path:
    out_path = Path(out_path)
    prs = Presentation()
    prs.slide_width = _SLIDE_W
    prs.slide_height = _SLIDE_H

    _title_slide(prs, title)
    for note in slides:
        _content_slide(prs, note)
    _admin_slides(prs, admin)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out_path))
    return out_path
