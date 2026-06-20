from __future__ import annotations

import os
import sys
from pathlib import Path

from danmaku.models import AppSettings


def resource_path(relative_path: str) -> Path:
    """
    Return an absolute path for bundled files.

    Works both in normal Python execution and after PyInstaller packaging.
    """
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / relative_path
    return Path.cwd() / relative_path


def load_text_file(path: Path, default: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return default


def load_settings_from_env() -> AppSettings:
    """
    Load basic settings from environment variables.

    Do not hardcode real API keys in the source code.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    use_dummy_raw = os.getenv("DANMAKU_USE_DUMMY_API", "true").strip().lower()

    capture_mode = os.getenv("CAPTURE_MODE", "full_screen").strip().lower()
    if capture_mode not in {"full_screen", "window", "region"}:
        capture_mode = "full_screen"

    capture_region = (0, 0, 0, 0)
    raw_capture_region = os.getenv("CAPTURE_REGION", "").strip()
    if raw_capture_region:
        parts = [part.strip() for part in raw_capture_region.split(",")]
        if len(parts) == 4:
            try:
                left, top, width, height = [int(part) for part in parts]
                if width > 0 and height > 0:
                    capture_region = (left, top, width, height)
                else:
                    capture_mode = "full_screen"
            except ValueError:
                capture_mode = "full_screen"

    return AppSettings(
        capture_interval_seconds=int(
            os.getenv("CAPTURE_INTERVAL_SECONDS", "6")),
        model_name=os.getenv("MODEL_NAME", "gemini-2.5-flash-lite"),
        api_key=api_key,
        use_dummy_api=use_dummy_raw in {
            "1", "true", "yes", "y"} or not api_key,
        capture_mode=capture_mode,
        capture_region=capture_region,
    )
