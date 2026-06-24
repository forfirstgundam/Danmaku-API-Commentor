from __future__ import annotations

import os
import sys
from pathlib import Path

from danmaku.models import AppSettings


API_KEY_ENV_BY_PROVIDER = {
    "anthropic": "ANTHROPIC_API_KEY",
    "deepinfra": "DEEPINFRA_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "openai": "OPENAI_API_KEY",
    "together": "TOGETHER_API_KEY",
    "xai": "XAI_API_KEY",
}

DEFAULT_MODEL_BY_PROVIDER = {
    "anthropic": "claude-haiku-4-5",
    "deepinfra": "Qwen/Qwen2.5-VL-7B-Instruct",
    "gemini": "gemini-3.1-flash-lite",
    "groq": "meta-llama/llama-4-scout-17b-16e-instruct",
    "mistral": "ministral-8b-2512",
    "openai": "gpt-5.4-mini",
    "together": "Qwen/Qwen3.5-9B",
    "xai": "grok-4.3",
}

DEFAULT_FALLBACK_MODEL_BY_PROVIDER = {
    "anthropic": "",
    "deepinfra": "",
    "gemini": "gemini-3.5-flash",
    "groq": "qwen/qwen3.6-27b",
    "mistral": "",
    "openai": "gpt-5.4-nano",
    "together": "",
    "xai": "",
}


def api_key_env_for_provider(provider: str) -> str:
    return API_KEY_ENV_BY_PROVIDER.get(provider, "GEMINI_API_KEY")


def default_model_for_provider(provider: str) -> str:
    return DEFAULT_MODEL_BY_PROVIDER.get(
        provider,
        DEFAULT_MODEL_BY_PROVIDER["gemini"],
    )


def default_fallback_model_for_provider(provider: str) -> str:
    return DEFAULT_FALLBACK_MODEL_BY_PROVIDER.get(provider, "")


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
    api_key_env = api_key_env_for_provider(api_provider)
    api_key = os.getenv(api_key_env, "").strip()
    default_model = default_model_for_provider(api_provider)
    default_fallback_model = default_fallback_model_for_provider(
        api_provider
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
    api_image_max_dimension = int(
        os.getenv("API_IMAGE_MAX_DIMENSION", "768")
    )
    configured_capture_format = os.getenv(
        "SAMPLE_CAPTURE_FORMAT",
        "JPEG",
    ).strip().upper()
    sample_capture_format = (
        "PNG"
        if api_image_max_dimension <= 0 or configured_capture_format == "PNG"
        else "JPEG"
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
        sample_capture_format=sample_capture_format,
        sample_capture_jpeg_quality=int(
            os.getenv("SAMPLE_CAPTURE_JPEG_QUALITY", "82")),
        api_provider=api_provider,
        model_name=os.getenv("MODEL_NAME", default_model),
        fallback_model_name=os.getenv(
            "FALLBACK_MODEL_NAME",
            default_fallback_model,
        ),
        api_key=api_key,
        send_screenshot_to_api=send_screenshot_raw in {"1", "true", "yes", "y"},
        api_image_max_dimension=api_image_max_dimension,
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
    )
