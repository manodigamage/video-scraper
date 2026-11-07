"""lecturenotes — turn lecture recordings into comprehensive PPTX notes.

Pipeline: ingest -> transcribe (local Whisper) -> frame dedup ->
vision extraction (Claude) -> align -> synthesize (Claude) -> build pptx.
"""

__version__ = "0.1.0"
