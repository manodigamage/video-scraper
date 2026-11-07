"""Optional reference materials: the lecturer's actual slides/notes (PPTX or PDF).

When provided, these give us *authoritative* text and crisp slide images, so
we don't have to rely on OCR of blurry video frames. We match each detected
video keyframe to the best source slide/page by text overlap; the match (if
confident) supplies clean printed text and a clean image for that slide, while
handwritten annotations still come from the video frame.

This is entirely optional — the pipeline runs normally when no materials are
given.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_WORD = re.compile(r"[a-z0-9]+")


@dataclass
class MaterialSlide:
    index: int
    text: str
    image_path: Optional[str] = None
    _tokens: frozenset[str] = field(default_factory=frozenset, repr=False)


@dataclass
class Materials:
    kind: str                     # "pptx" | "pdf"
    slides: list[MaterialSlide]


def _tokenize(text: str) -> frozenset[str]:
    return frozenset(w for w in _WORD.findall(text.lower()) if len(w) > 2)


# ---------- loaders ----------

def _import_pymupdf():
    try:
        import pymupdf  # type: ignore
        return pymupdf
    except ImportError:
        import fitz  # type: ignore  # older PyMuPDF
        return fitz


def _render_pdf_pages(pdf_path: Path, out_dir: Path, dpi: int = 150) -> list[tuple[str, Optional[str]]]:
    """Return (text, image_path) per page using PyMuPDF."""
    fitz = _import_pymupdf()
    out_dir.mkdir(parents=True, exist_ok=True)
    pages: list[tuple[str, Optional[str]]] = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text")
            img = out_dir / f"src_{i:03d}.png"
            try:
                page.get_pixmap(dpi=dpi).save(str(img))
                image_path: Optional[str] = str(img)
            except Exception:
                image_path = None
            pages.append((text, image_path))
    return pages


def load_pdf(path: Path, work: Path) -> Materials:
    pages = _render_pdf_pages(path, work / "materials")
    slides = [MaterialSlide(index=i, text=t, image_path=img, _tokens=_tokenize(t))
              for i, (t, img) in enumerate(pages)]
    return Materials(kind="pdf", slides=slides)


def load_pptx(path: Path, work: Path) -> Materials:
    from pptx import Presentation  # type: ignore

    prs = Presentation(str(path))
    texts: list[str] = []
    for slide in prs.slides:
        parts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                parts.append(shape.text_frame.text)
        texts.append("\n".join(p for p in parts if p.strip()))

    # Try to render crisp slide images via LibreOffice -> PDF -> PyMuPDF.
    images: list[Optional[str]] = [None] * len(texts)
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice:
        try:
            work.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [soffice, "--headless", "--convert-to", "pdf", "--outdir", str(work), str(path)],
                capture_output=True, text=True, timeout=180, check=True)
            pdf = work / (path.stem + ".pdf")
            if pdf.exists():
                rendered = _render_pdf_pages(pdf, work / "materials")
                for i, (_t, img) in enumerate(rendered):
                    if i < len(images):
                        images[i] = img
        except Exception:
            pass  # fall back to text-only; video frames still embed

    slides = [MaterialSlide(index=i, text=t, image_path=images[i], _tokens=_tokenize(t))
              for i, t in enumerate(texts)]
    return Materials(kind="pptx", slides=slides)


def load_materials(path: str | Path, work: str | Path) -> Optional[Materials]:
    path = Path(path)
    work = Path(work)
    if not path.exists():
        raise FileNotFoundError(f"Materials file not found: {path}")
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path, work)
    if ext in (".pptx", ".ppt"):
        return load_pptx(path, work)
    raise ValueError(f"Unsupported materials type: {ext} (use .pdf or .pptx)")


# ---------- matching ----------

def best_match(printed_text: str, materials: Materials, threshold: float = 0.30
               ) -> tuple[Optional[MaterialSlide], float]:
    """Match a video frame's printed text to the most similar source slide.

    Score = overlap of the frame's tokens that appear in the source slide
    (coverage of the frame by the slide). Returns (slide, score) or (None, 0).
    """
    q = _tokenize(printed_text)
    if not q:
        return None, 0.0
    best: Optional[MaterialSlide] = None
    best_score = 0.0
    for s in materials.slides:
        if not s._tokens:
            continue
        score = len(q & s._tokens) / len(q)
        if score > best_score:
            best, best_score = s, score
    return (best, best_score) if best_score >= threshold else (None, best_score)
