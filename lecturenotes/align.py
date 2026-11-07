"""Stage 5: align the transcript to each visual state by timestamp.

Each keyframe owns the interval [its timestamp, next keyframe's timestamp).
Spoken text is assigned to the interval its midpoint falls into, so no spoken
content is dropped. Speech before the first keyframe is attached to the first;
speech after the last keyframe is attached to the last.
"""
from __future__ import annotations

from .models import FrameExtraction, FrameKind, Keyframe, Section, TranscriptSegment


def build_sections(
    keyframes: list[Keyframe],
    extractions: dict[int, FrameExtraction],
    transcript: list[TranscriptSegment],
) -> list[Section]:
    if not keyframes:
        # no visuals at all -> one section holding the whole transcript
        text = " ".join(s.text for s in transcript)
        return [Section(
            keyframe=Keyframe(index=0, timestamp=0.0, image_path=""),
            extraction=FrameExtraction(frame_index=0, kind=FrameKind.OTHER),
            spoken_text=text, start=0.0,
            end=transcript[-1].end if transcript else 0.0,
        )]

    bounds: list[tuple[float, float]] = []
    for i, kf in enumerate(keyframes):
        start = kf.timestamp
        end = keyframes[i + 1].timestamp if i + 1 < len(keyframes) else float("inf")
        bounds.append((start, end))

    buckets: dict[int, list[str]] = {kf.index: [] for kf in keyframes}
    for seg in transcript:
        mid = (seg.start + seg.end) / 2.0
        placed = False
        for i, (start, end) in enumerate(bounds):
            if start <= mid < end:
                buckets[keyframes[i].index].append(seg.text)
                placed = True
                break
        if not placed:
            # before first keyframe
            if mid < bounds[0][0]:
                buckets[keyframes[0].index].append(seg.text)
            else:
                buckets[keyframes[-1].index].append(seg.text)

    sections: list[Section] = []
    for i, kf in enumerate(keyframes):
        sections.append(Section(
            keyframe=kf,
            extraction=extractions.get(kf.index, FrameExtraction(
                frame_index=kf.index, kind=FrameKind.OTHER)),
            spoken_text=" ".join(buckets[kf.index]).strip(),
            start=bounds[i][0],
            end=keyframes[i + 1].timestamp if i + 1 < len(keyframes) else kf.timestamp,
        ))
    return sections
