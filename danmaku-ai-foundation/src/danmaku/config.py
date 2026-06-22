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
    api_provider = os.getenv("API_PROVIDER", "gemini").strip().lower()
    api_key_env = "OPENAI_API_KEY" if api_provider == "openai" else "GEMINI_API_KEY"
    api_key = os.getenv(api_key_env, "").strip()
    default_model = (
        "gpt-5.4-nano"
        if api_provider == "openai"
        else "gemini-2.5-flash-lite"
    )
    use_dummy_raw = os.getenv("DANMAKU_USE_DUMMY_API", "true").strip().lower()
    send_screenshot_raw = os.getenv(
        "DANMAKU_SEND_SCREENSHOT", "true").strip().lower()
    save_api_images_raw = os.getenv(
        "DANMAKU_SAVE_API_IMAGES", "true").strip().lower()
    use_streaming_raw = os.getenv(
        "DANMAKU_USE_STREAMING_API", "true").strip().lower()
    use_multi_frame_raw = os.getenv(
        "DANMAKU_USE_MULTI_FRAME", "true").strip().lower()
    use_ocr_raw = os.getenv(
        "DANMAKU_USE_OCR", "false").strip().lower()
    ocr_languages = tuple(
        item.strip()
        for item in os.getenv("DANMAKU_OCR_LANGUAGES", "ko,en").split(",")
        if item.strip()
    )

    return AppSettings(
        capture_interval_seconds=int(
            os.getenv("CAPTURE_INTERVAL_SECONDS", "6")),
        use_multi_frame_context=use_multi_frame_raw in {
            "1", "true", "yes", "y"},
        frame_sample_interval_seconds=int(
            os.getenv("FRAME_SAMPLE_INTERVAL_SECONDS", "1")),
        frame_buffer_size=int(os.getenv("FRAME_BUFFER_SIZE", "6")),
        frames_per_request=int(os.getenv("FRAMES_PER_REQUEST", "4")),
        sample_capture_jpeg_quality=int(
            os.getenv("SAMPLE_CAPTURE_JPEG_QUALITY", "82")),
        api_provider=api_provider,
        model_name=os.getenv("MODEL_NAME", default_model),
        fallback_model_name=os.getenv("FALLBACK_MODEL_NAME", "gemini-3.5-flash"),
        api_key=api_key,
        send_screenshot_to_api=send_screenshot_raw in {"1", "true", "yes", "y"},
        api_image_max_dimension=int(os.getenv("API_IMAGE_MAX_DIMENSION", "768")),
        api_image_jpeg_quality=int(os.getenv("API_IMAGE_JPEG_QUALITY", "72")),
        history_image_max_dimension=int(
            os.getenv("HISTORY_IMAGE_MAX_DIMENSION", "384")),
        history_image_jpeg_quality=int(
            os.getenv("HISTORY_IMAGE_JPEG_QUALITY", "42")),
        api_max_output_tokens=int(os.getenv("API_MAX_OUTPUT_TOKENS", "512")),
        use_streaming_api=use_streaming_raw in {"1", "true", "yes", "y"},
        save_api_images=save_api_images_raw in {"1", "true", "yes", "y"},
        use_dummy_api=use_dummy_raw in {
            "1", "true", "yes", "y"} or not api_key,
        ocr_enabled=use_ocr_raw in {"1", "true", "yes", "y"},
        ocr_languages=ocr_languages or ("ko", "en"),
        ocr_capture_interval_ms=int(
            os.getenv("DANMAKU_OCR_INTERVAL_MS", "500")
        ),
    )
