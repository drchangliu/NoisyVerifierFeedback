"""Shared loader for configs/model_registry.json.

The registry lets a new model join the analysis pipeline and leaderboard
without code edits: tau_star_prediction, make_leaderboard, run_humaneval,
and magazine_figure all merge it into their static tables at import time.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = ROOT / "configs" / "model_registry.json"

# Colors for vendors not in the original six-vendor palette.
EXTRA_VENDOR_COLORS = {
    "Moonshot": "#795548",
    "OpenAI": "#c2255c",
    "MiniMax": "#e8590c",
    "Nvidia": "#0c8599",
}
FALLBACK_VENDOR_COLOR = "#7f7f7f"


def load_registry() -> dict:
    """Return {model_tag: entry} for real entries (comment keys stripped)."""
    if not REGISTRY_PATH.exists():
        return {}
    data = json.loads(REGISTRY_PATH.read_text())
    return {k: v for k, v in data.items() if not k.startswith("_")}
