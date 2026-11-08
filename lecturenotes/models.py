"""Shared data types passed between pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class FrameKind(str, Enum):
    SLIDE = "slide"              # printed presentation slide
    ANNOTATED_SLIDE = "annotated_slide"  # slide with lecturer's handwriting on top
    WHITEBOARD = "whiteboard"    # physical/shared board the lecturer writes on
    OTHER = "other"              # talking head, logo, blank, transition -> usually dropped


@dataclass
class TranscriptSegment:
    start: float   # seconds
    end: float
    text: str


@dataclass
class Keyframe:
    index: int
    timestamp: float            # seconds into the video
    image_path: str
    phash: str = ""
    # filled by vision stage:
    end_timestamp: Optional[float] = None  # when this visual state is replaced


@dataclass
class Annotation:
    """One thing the lecturer added on top of a slide/board."""
    refers_to: str   # short description of which region/topic it relates to
    text: str        # cleaned, legible transcription of what was written/drawn
    uncertain: bool = False


@dataclass
class FrameExtraction:
    frame_index: int
    kind: FrameKind
    printed_text: str = ""              # OCR of printed slide text
    handwriting_text: str = ""          # transcription of board/handwritten content
    diagram_description: str = ""       # words describing any drawing/diagram
    annotations: list[Annotation] = field(default_factory=list)
    unclear: bool = False               # model was unsure about some content
    model_used: str = ""


@dataclass
class Section:
    """A visual state plus everything spoken while it was on screen."""
    keyframe: Keyframe
    extraction: FrameExtraction
    spoken_text: str
    start: float
    end: float
    # optional: authoritative source slide/page matched from uploaded materials
    source_text: str = ""            # clean text from the lecturer's own slide/notes
    source_image_path: Optional[str] = None  # crisp source slide image, if available


@dataclass
class SlideNote:
    title: str
    bullets: list[str] = field(default_factory=list)
    continued: bool = False             # True => "(cont.)" overflow slide
    image_path: Optional[str] = None    # keyframe image to embed, if any
    note: str = ""                      # optional free-text block (e.g. board diagram desc)


@dataclass
class AdminItem:
    category: str   # assignment | exam | attendance | submission | deadline | other
    text: str
    timestamp: float


@dataclass
class LectureResult:
    title: str
    slides: list[SlideNote]
    admin: list[AdminItem]
    transcript: list[TranscriptSegment]
