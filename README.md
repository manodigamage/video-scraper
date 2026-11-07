# lecture-notes-tool

Turn a lecture recording into a comprehensive PowerPoint of study notes —
capturing the slides and whiteboard (including the lecturer's handwriting and
annotations) **merged with** what they actually said, plus an appendix of
announcements (assignments, exams, attendance, deadlines).

Runs locally on your machine. Transcription happens on your **NVIDIA GPU** for
free; a small number of calls to the **Claude API** read the on-screen content
and write the notes.

> Status: **Phase 1 — core CLI pipeline.** The native desktop app (PySide6)
> comes in Phase 3. Code is written but not yet validated on real hardware —
> the first run on a sample lecture is the calibration step.

---

## How it works

```
video ──► extract audio ──► transcribe (local Whisper, GPU)
      └──► sample frames ──► dedup to distinct slides/board states
                                  │
                   Claude reads each state (OCR + handwriting + diagrams)
                                  │
             align transcript to each state by timestamp
                                  │
        Claude merges "shown + said" → clean slide notes (+ overflow split)
                                  │
             scan transcript for admin announcements → appendix
                                  │
                         build  notes.pptx
```

Design rationale, cost model, and roadmap live in the project spec
(`lecture-tool-spec.md`).

---

## Requirements

- **Python 3.11+**
- **ffmpeg** on your PATH (install via your OS package manager, not pip)
- An **NVIDIA GPU** with recent drivers (for fast local transcription; it falls
  back to CPU, just slower)
- An **Anthropic API key** with a little prepaid credit —
  [console.anthropic.com](https://console.anthropic.com). This is **separate
  from a Claude.ai subscription**; $5 of credit covers dozens of lectures.

## Install

```bash
git clone https://github.com/manodigamage/lecture-notes-tool.git
cd lecture-notes-tool
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# NVIDIA CUDA libs for faster-whisper (if not already present):
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

## Set your API key

```bash
# option A: environment variable
export ANTHROPIC_API_KEY=sk-ant-...
# option B: store securely in the OS keyring
python -m lecturenotes.cli --save-key --api-key sk-ant-...
```

## Run

```bash
python -m lecturenotes.cli my_lecture.mp4 -o my_lecture.pptx
python -m lecturenotes.cli my_lecture.mp4 --cheap     # Haiku only (cheapest)
```

Intermediate files (full transcript, extracted keyframes, notes JSON) are kept
next to the output for inspection.

## Cost & speed (targets)

- **Cost:** ~$0.05–0.20 per lecture (Haiku workhorse + optional Sonnet on hard
  frames). Verify current model prices in the Console.
- **Speed:** target ≤ 30 min for a 2–3 hr video on an NVIDIA GPU; dominated by
  transcription.

## Known limitations (v1)

- Annotated slides keep the **original slide image** with the lecturer's
  additions as attached, legible notes — not a pixel-perfect redraw onto the
  slide.
- Genuinely illegible handwriting is given a best-guess transcription marked
  `[unclear]`.
- Complex diagrams are described in words and kept as the cropped image, not
  vectorized.

## License

MIT (add a LICENSE file).
