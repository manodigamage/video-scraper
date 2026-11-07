"""Stage 7: render slide notes + admin appendix into a polished .pptx (FR11).

The deck is authored natively (not a patched screenshot): each content slide
has a styled title, an accent bar, the embedded lecture image, and clean notes.
All colors live in THEME so the look is easy to retune.
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

from .models import AdminItem, SlideNote

_SLIDE_W = Inches(13.333)
_SLIDE_H = Inches(7.5)


class THEME:
    PRIMARY = RGBColor(0x4F, 0x46, 0xE5)       # indigo accent
    PRIMARY_DARK = RGBColor(0x31, 0x2E, 0x81)
    ADMIN = RGBColor(0xD9, 0x77, 0x06)         # amber accent for logistics
    ADMIN_DARK = RGBColor(0x92, 0x4D, 0x05)
    INK = RGBColor(0x1F, 0x29, 0x37)           # body text
    MUTED = RGBColor(0x6B, 0x72, 0x80)         # secondary text
    HAIRLINE = RGBColor(0xE5, 0xE7, 0xEB)
    WHITE = RGBColor(0xFF, 0xFF, 0xFF)
    TITLE_BG = RGBColor(0xF5, 0xF3, 0xFF)      # faint tint for title slide


def _blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def _rect(slide, left, top, width, height, color, line=None):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(0.75)
    shp.shadow.inherit = False
    return shp


def _text(slide, left, top, width, height, runs, align=PP_ALIGN.LEFT,
          anchor=MSO_ANCHOR.TOP):
    """runs: list of paragraphs; each paragraph is list of (text,size,color,bold)."""
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for i, para in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(6)
        for (t, size, color, bold) in para:
            r = p.add_run()
            r.text = t
            r.font.size = Pt(size)
            r.font.color.rgb = color
            r.font.bold = bold
    return box


def _accent_bar(slide, color):
    _rect(slide, 0, 0, Inches(0.22), _SLIDE_H, color)


def _footer(slide, title, number):
    _text(slide, Inches(0.5), Inches(7.02), Inches(9), Inches(0.4),
          [[(title, 9, THEME.MUTED, False)]])
    _text(slide, Inches(12.0), Inches(7.02), Inches(1.0), Inches(0.4),
          [[(str(number), 9, THEME.MUTED, False)]], align=PP_ALIGN.RIGHT)


def _title_slide(prs, title):
    slide = _blank(prs)
    _rect(slide, 0, 0, _SLIDE_W, _SLIDE_H, THEME.TITLE_BG)
    _rect(slide, 0, Inches(2.55), _SLIDE_W, Inches(0.12), THEME.PRIMARY)
    _text(slide, Inches(1.0), Inches(2.75), Inches(11.3), Inches(1.4),
          [[(title, 44, THEME.PRIMARY_DARK, True)]], anchor=MSO_ANCHOR.MIDDLE)
    _text(slide, Inches(1.0), Inches(4.15), Inches(11.3), Inches(0.6),
          [[("Lecture notes — generated from the recording", 18, THEME.MUTED, False)]])


def _bullets_para(bullets):
    return [[("•  " + b, 16, THEME.INK, False)] for b in bullets]


def _content_slide(prs, note, title, number):
    slide = _blank(prs)
    _accent_bar(slide, THEME.PRIMARY)

    heading = note.title + "  (cont.)" if note.continued else note.title
    _text(slide, Inches(0.55), Inches(0.35), Inches(12.3), Inches(0.9),
          [[(heading, 28, THEME.PRIMARY_DARK, True)]])
    _rect(slide, Inches(0.6), Inches(1.2), Inches(3.2), Pt(3), THEME.PRIMARY)

    has_image = bool(note.image_path) and Path(note.image_path).exists()
    if has_image:
        try:
            pic = slide.shapes.add_picture(
                note.image_path, Inches(0.55), Inches(1.5), width=Inches(6.5))
            pic.line.color.rgb = THEME.HAIRLINE
            pic.line.width = Pt(1)
        except Exception:
            has_image = False

    if has_image:
        _text(slide, Inches(7.35), Inches(1.45), Inches(5.45), Inches(5.3),
              _bullets_para(note.bullets))
        if note.note:
            _text(slide, Inches(0.55), Inches(6.35), Inches(6.5), Inches(0.6),
                  [[(note.note, 12, THEME.MUTED, False)]])
    else:
        _text(slide, Inches(0.6), Inches(1.45), Inches(12.1), Inches(5.3),
              _bullets_para(note.bullets))
        if note.note:
            _text(slide, Inches(0.6), Inches(6.35), Inches(12.1), Inches(0.5),
                  [[(note.note, 12, THEME.MUTED, False)]])
    _footer(slide, title, number)


def _admin_slides(prs, admin, title, start_number):
    if not admin:
        return start_number
    by_cat: dict[str, list[str]] = {}
    for it in admin:
        by_cat.setdefault(it.category, []).append(it.text)

    paras: list[list[tuple]] = []
    for cat, items in by_cat.items():
        paras.append([(cat.upper(), 15, THEME.ADMIN_DARK, True)])
        for it in items:
            paras.append([("•  " + it, 15, THEME.INK, False)])

    number = start_number
    chunk = 11
    first = True
    for i in range(0, len(paras), chunk):
        slide = _blank(prs)
        _accent_bar(slide, THEME.ADMIN)
        head = "Announcements & Admin" + ("  (cont.)" if not first else "")
        first = False
        _text(slide, Inches(0.55), Inches(0.35), Inches(12.3), Inches(0.9),
              [[(head, 28, THEME.ADMIN_DARK, True)]])
        _rect(slide, Inches(0.6), Inches(1.2), Inches(3.2), Pt(3), THEME.ADMIN)
        _text(slide, Inches(0.6), Inches(1.5), Inches(12.1), Inches(5.3),
              paras[i:i + chunk])
        _footer(slide, title, number)
        number += 1
    return number


def build_pptx(title, slides, admin, out_path):
    out_path = Path(out_path)
    prs = Presentation()
    prs.slide_width = _SLIDE_W
    prs.slide_height = _SLIDE_H

    _title_slide(prs, title)
    n = 1
    for note in slides:
        _content_slide(prs, note, title, n)
        n += 1
    _admin_slides(prs, admin, title, n)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out_path))
    return out_path
