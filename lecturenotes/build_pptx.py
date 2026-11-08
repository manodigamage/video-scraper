"""Stage 7: render slide notes + admin appendix into a clean .pptx (FR11).

Priorities: clarity and readability. No decorative accent bar, no overlapping
footer. Content is paginated deterministically so nothing overflows the slide,
and tabular data (stats / parameter grids) renders as a real table.
"""
import math
import tempfile
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

from .models import AdminItem, SlideNote

_SLIDE_W = Inches(13.333)
_SLIDE_H = Inches(7.5)


class THEME:
    PRIMARY = RGBColor(0x31, 0x2E, 0x81)       # title indigo
    ADMIN = RGBColor(0x92, 0x4D, 0x05)         # admin amber
    INK = RGBColor(0x1F, 0x29, 0x37)
    MUTED = RGBColor(0x6B, 0x72, 0x80)
    HAIRLINE = RGBColor(0xD1, 0xD5, 0xDB)
    HEADER_BG = RGBColor(0xEE, 0xED, 0xFE)
    TITLE_BG = RGBColor(0xF5, 0xF3, 0xFF)

# layout constants
_BODY_PT = 15
_NOTE_PT = 12
# rough chars-per-line and line-capacity for each column type (kept conservative)
_RIGHT_CPL, _RIGHT_LINES = 46, 15          # notes column beside an image
_FULL_CPL, _FULL_LINES = 105, 18           # full-width notes column


def _blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def _para(tf, runs, first, size, color, bold=False, space=6):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.space_after = Pt(space)
    r = p.add_run()
    r.text = runs
    r.font.size = Pt(size)
    r.font.color.rgb = color
    r.font.bold = bold
    return p


def _title(slide, text, continued=False, color=None):
    color = color or THEME.PRIMARY
    box = slide.shapes.add_textbox(Inches(0.55), Inches(0.35), Inches(12.3), Inches(0.95))
    tf = box.text_frame
    tf.word_wrap = True
    _para(tf, text + ("  (cont.)" if continued else ""), True, 26, color, bold=True)


def _prep_image(path):
    try:
        from PIL import Image, ImageChops
        im = Image.open(path).convert("RGB")
        gray = im.convert("L")
        diff = ImageChops.difference(gray, Image.new("L", gray.size, 0))
        bbox = diff.point(lambda p: 255 if p > 18 else 0).getbbox()
        if bbox and (bbox[2] - bbox[0]) > im.width * 0.3 \
                and (bbox[3] - bbox[1]) > im.height * 0.3:
            out = Path(tempfile.gettempdir()) / (Path(path).stem + "_trim.jpg")
            im.crop(bbox).save(out, quality=88)
            return str(out)
    except Exception:
        pass
    return path


def _lines_for(text, cpl):
    return max(1, math.ceil(len(text) / cpl))


def _paginate(items, has_image):
    """items: list of (text, muted). Split into pages that fit. The first page
    uses the narrower notes column when an image is present; the rest are full
    width."""
    pages = []
    idx = 0
    first = True
    while idx < len(items):
        if first and has_image:
            cpl, cap = _RIGHT_CPL, _RIGHT_LINES
        else:
            cpl, cap = _FULL_CPL, _FULL_LINES
        used, page = 0, []
        while idx < len(items):
            text, muted = items[idx]
            need = _lines_for(text, cpl) + 1          # +1 for inter-bullet gap
            if page and used + need > cap:
                break
            page.append(items[idx])
            used += need
            idx += 1
        pages.append(page)
        first = False
    return pages or [[]]


def _bullets_box(slide, items, left, top, width, height):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    first = True
    for text, muted in items:
        _para(tf, ("" if muted else "•  ") + text, first,
              _NOTE_PT if muted else _BODY_PT,
              THEME.MUTED if muted else THEME.INK)
        first = False


def _add_table(slide, rows, left, top, width):
    nrows = len(rows)
    ncols = max(len(r) for r in rows)
    height = Inches(0.34 * nrows)
    gfx = slide.shapes.add_table(nrows, ncols, left, top, width, height)
    tbl = gfx.table
    for i, row in enumerate(rows):
        for j in range(ncols):
            cell = tbl.cell(i, j)
            cell.text = str(row[j]) if j < len(row) else ""
            cell.margin_top = Pt(2)
            cell.margin_bottom = Pt(2)
            for p in cell.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(12)
                    r.font.bold = (i == 0)
                    r.font.color.rgb = THEME.PRIMARY if i == 0 else THEME.INK
    return 0.34 * nrows


def _title_slide(prs, title):
    slide = _blank(prs)
    bg = slide.shapes.add_shape  # fill background via a rectangle
    from pptx.enum.shapes import MSO_SHAPE
    rect = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, _SLIDE_W, _SLIDE_H)
    rect.fill.solid(); rect.fill.fore_color.rgb = THEME.TITLE_BG
    rect.line.fill.background(); rect.shadow.inherit = False
    box = slide.shapes.add_textbox(Inches(1.0), Inches(2.9), Inches(11.3), Inches(1.6))
    tf = box.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    _para(tf, title, True, 40, THEME.PRIMARY, bold=True)
    _para(tf, "Lecture notes — generated from the recording", False, 16, THEME.MUTED)


def _content_slides(prs, note: SlideNote):
    """Emit one or more slides for a single note, paginating to avoid overflow."""
    items = [(b, False) for b in note.bullets if b.strip()]
    if note.note:
        items.append((note.note, True))

    has_image = bool(note.image_path) and Path(note.image_path).exists()
    img = _prep_image(note.image_path) if has_image else None

    # table (if any) lives on the first slide; bullets paginate around it
    pages = _paginate(items, has_image)
    for pi, page in enumerate(pages):
        slide = _blank(prs)
        _title(slide, note.title, continued=(pi > 0 or note.continued))

        first_page = (pi == 0)
        if first_page and has_image:
            try:
                slide.shapes.add_picture(img, Inches(0.55), Inches(1.5), width=Inches(6.5))
            except Exception:
                has_image = False
            nx, nw = Inches(7.35), Inches(5.45)
        else:
            nx, nw = Inches(0.6), Inches(12.1)
        ntop = Inches(1.45)

        if first_page and note.table:
            used_in = _add_table(slide, note.table, nx, ntop, nw)
            ntop = Inches(1.45 + used_in + 0.25)

        _bullets_box(slide, page, nx, ntop, nw, Inches(7.0) - ntop)


def _admin_slides(prs, admin):
    if not admin:
        return
    by_cat = {}
    for it in admin:
        by_cat.setdefault(it.category, []).append(it.text)
    items = []
    for cat, rows in by_cat.items():
        items.append((cat.upper(), True))
        items.extend((r, False) for r in rows)

    pages = _paginate(items, has_image=False)
    for pi, page in enumerate(pages):
        slide = _blank(prs)
        _title(slide, "Announcements & Admin", continued=(pi > 0), color=THEME.ADMIN)
        # render: muted items are category headers (amber, bold)
        box = slide.shapes.add_textbox(Inches(0.6), Inches(1.5), Inches(12.1), Inches(5.3))
        tf = box.text_frame; tf.word_wrap = True
        first = True
        for text, is_header in page:
            if is_header:
                _para(tf, text, first, 15, THEME.ADMIN, bold=True)
            else:
                _para(tf, "•  " + text, first, 15, THEME.INK)
            first = False


def build_pptx(title, slides, admin, out_path):
    out_path = Path(out_path)
    prs = Presentation()
    prs.slide_width = _SLIDE_W
    prs.slide_height = _SLIDE_H

    _title_slide(prs, title)
    for note in slides:
        _content_slides(prs, note)
    _admin_slides(prs, admin)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        prs.save(str(out_path))
        return out_path
    except PermissionError:
        # target is open/locked in another app (e.g. PowerPoint) — never lose
        # the whole run; save under a timestamped name instead.
        import time
        alt = out_path.with_name(
            f"{out_path.stem}-{time.strftime('%Y%m%d-%H%M%S')}{out_path.suffix}")
        prs.save(str(alt))
        return alt
