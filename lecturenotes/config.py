"""Configuration and API-key handling.

Resolution order for the Anthropic API key:
  1. explicit argument
  2. ANTHROPIC_API_KEY environment variable
  3. OS keyring (service "lecture-notes-tool"), if `keyring` is installed

Other settings have sensible defaults and can be overridden via a TOML
file (see config.example.toml) passed to `load_config`.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# NOTE: verify/update these model IDs against the Anthropic Console — model
# names change between releases. Haiku is the cheap workhorse; Sonnet is used
# only for frames flagged as hard (unclear handwriting / dense boards).
DEFAULT_WORKHORSE_MODEL = "claude-haiku-5-5"
DEFAULT_ESCALATION_MODEL = "claude-sonnet-5-5"

KEYRING_SERVICE = "lecture-notes-tool"
KEYRING_USER = "anthropic-api-key"


@dataclass
class Config:
    # --- API ---
    api_key: str = ""
    workhorse_model: str = DEFAULT_WORKHORSE_MODEL
    escalation_model: str = DEFAULT_ESCALATION_MODEL
    allow_escalation: bool = True       # "quality mode"; False = Haiku only (cheapest)
    max_api_concurrency: int = 8        # parallel API calls

    # --- transcription ---
    whisper_model: Optional[str] = None  # None => auto-pick by VRAM
    whisper_language: Optional[str] = None  # None => autodetect
    compute_type: str = "float16"       # faster-whisper compute type on GPU

    # --- frame sampling / dedup ---
    sample_fps: float = 1.0             # how often to look at the video (frames/sec)
    phash_distance_threshold: int = 6   # <= this hamming distance => "same" visual state
    min_state_seconds: float = 3.0      # ignore flickers shorter than this
    image_max_dim: int = 1568           # longest edge sent to the vision model (px)

    # --- optional reference materials (lecturer's slides/notes) ---
    materials_match_threshold: float = 0.30  # min token-coverage to trust a slide match

    # --- misc ---
    workdir: Optional[str] = None       # scratch dir; None => temp dir next to output
    keep_intermediates: bool = True     # keep transcript + frames beside the pptx

    def require_api_key(self) -> str:
        if not self.api_key:
            raise RuntimeError(
                "No Anthropic API key found. Set ANTHROPIC_API_KEY, store it in the "
                "OS keyring, or pass it in Settings. Get one at console.anthropic.com "
                "(this is separate from a Claude.ai subscription)."
            )
        return self.api_key


def _key_from_keyring() -> str:
    try:
        import keyring  # type: ignore
        return keyring.get_password(KEYRING_SERVICE, KEYRING_USER) or ""
    except Exception:
        return ""


def resolve_api_key(explicit: Optional[str] = None) -> str:
    return explicit or os.environ.get("ANTHROPIC_API_KEY", "") or _key_from_keyring()


def save_api_key(key: str) -> bool:
    """Persist the key in the OS keyring. Returns True on success."""
    try:
        import keyring  # type: ignore
        keyring.set_password(KEYRING_SERVICE, KEYRING_USER, key)
        return True
    except Exception:
        return False


def load_config(path: Optional[str | Path] = None, api_key: Optional[str] = None) -> Config:
    cfg = Config()
    if path:
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        for section in data.values() if isinstance(data, dict) else []:
            if isinstance(section, dict):
                for k, v in section.items():
                    if hasattr(cfg, k):
                        setattr(cfg, k, v)
        # also allow flat keys
        for k, v in data.items():
            if hasattr(cfg, k) and not isinstance(v, dict):
                setattr(cfg, k, v)
    cfg.api_key = resolve_api_key(api_key)
    return cfg
