"""Central config for v2 ingest scripts.

Loads .env from project root and exposes the minimal set of constants the
scripts in scripts/ actually use. Path layout mirrors v1 so the scripts
themselves stay close to their v1-equivalents in shape, but everything
points at v2's own tree.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _required(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Qdrant Cloud — shared cluster with v1. ALL v2 collection names must start
# with "v2_" — enforced by qdrant_safe.SafeQdrantClient at runtime.
QDRANT_URL = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")

# v2 collection names (the v2_ prefix is mandatory; see qdrant_safe.py)
QDRANT_COLLECTION_FRAMES = "v2_frames"   # CLIP image vectors, one per keyframe

ALEX_REFERENCE_AUDIO = os.environ.get("ALEX_REFERENCE_AUDIO", "").strip() or None
LEILA_REFERENCE_AUDIO = os.environ.get("LEILA_REFERENCE_AUDIO", "").strip() or None
SHARRAN_REFERENCE_AUDIO = os.environ.get("SHARRAN_REFERENCE_AUDIO", "").strip() or None

MEDIA_DIR = PROJECT_ROOT / "ingest" / "media"
CACHE_DIR = PROJECT_ROOT / "ingest" / "cache"
FRAMES_DIR = PROJECT_ROOT / "ingest" / "frames"
LOGS_DIR = PROJECT_ROOT / "ingest" / "logs"
