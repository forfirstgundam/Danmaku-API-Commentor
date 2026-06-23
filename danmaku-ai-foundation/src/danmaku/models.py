from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class SessionProfile:
    """Conservative structured interpretation of a user's stream label."""

    title: str = ""
    content_type: str = "unknown"
    activity: str = ""
    episode: str = ""
    subtitle_language: str = ""
    source_description: str = ""
    status: str = "empty"
    interpretation_error: str = ""

    @classmethod
    def pending(cls, description: str) -> "SessionProfile":
        return cls(
            source_description=description.strip(),
            status="pending" if description.strip() else "empty",
        )

    @classmethod
    def fallback(
        cls,
        description: str,
        error: str = "",
    ) -> "SessionProfile":
        return cls(
            source_description=description.strip(),
            status="fallback",
            interpretation_error=error.strip(),
        )

    @classmethod
    def from_mapping(
        cls,
        data: dict[str, object],
        description: str,
    ) -> "SessionProfile":
        allowed_types = {
            "unknown",
            "anime",
            "game",
            "video",
            "manga",
            "other",
        }

        def clean(key: str) -> str:
            value = data.get(key)
            return value.strip() if isinstance(value, str) else ""

        content_type = clean("content_type").lower()
        if content_type not in allowed_types:
            content_type = "unknown"

        return cls(
            title=clean("title"),
            content_type=content_type,
            activity=clean("activity"),
            episode=clean("episode"),
            subtitle_language=clean("subtitle_language"),
            source_description=description.strip(),
            status="interpreted",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title or None,
            "content_type": self.content_type,
            "activity": self.activity or None,
            "episode": self.episode or None,
            "subtitle_language": self.subtitle_language or None,
            "source_description": self.source_description,
            "status": self.status,
            "interpretation_error": self.interpretation_error or None,
        }

    def to_prompt_text(self) -> str:
        fields = [
            f"Title: {self.title or 'unknown'}",
            f"Content type: {self.content_type or 'unknown'}",
            f"Activity: {self.activity or 'unknown'}",
            f"Episode/chapter: {self.episode or 'unknown'}",
            (
                "Subtitle language: "
                f"{self.subtitle_language or 'unknown'}"
            ),
        ]
        return "\n".join(f"- {field}" for field in fields)


@dataclass(slots=True)
class AppSettings:
    """Runtime settings shared across modules."""
    first_capture_delay_ms: int = 1500

    capture_interval_seconds: int = 6
    use_multi_frame_context: bool = True
    frame_sample_interval_seconds: int = 1
    frame_buffer_size: int = 6
    frames_per_request: int = 4
    sample_capture_jpeg_quality: int = 82
    api_provider: str = "gemini"
    model_name: str = "gemini-2.5-flash-lite"
    fallback_model_name: str = "gemini-3.5-flash"
    api_key: str = ""
    use_dummy_api: bool = True
    send_screenshot_to_api: bool = True
    api_image_max_dimension: int = 768
    api_image_jpeg_quality: int = 72
    history_image_max_dimension: int = 384
    history_image_jpeg_quality: int = 42
    api_max_output_tokens: int = 512
    use_streaming_api: bool = True

    # Stable user-provided and interpreted context for this viewing session.
    user_stream_description: str = ""
    session_profile: SessionProfile = field(default_factory=SessionProfile)

    # Testing / logging
    save_captures: bool = True
    save_comments: bool = True
    save_api_images: bool = True

    # Base folder for all runs
    log_root_dir: Path = Path("logs")

    # These are filled/replaced when Start is pressed
    run_log_dir: Path = Path("logs/current")
    capture_output_dir: Path = Path("logs/current/captures")
    api_image_output_dir: Path = Path("logs/current/api_images")
    comment_log_path: Path = Path("logs/current/comments.jsonl")

    # Capture settings
    target_window_title: str = ""  # empty means full screen
    target_window_handle: int = 0  # stable Windows HWND selected in the UI
    capture_mode: str = "full_screen"
    capture_region: tuple[int, int, int, int] = (0, 0, 0, 0)
    prompt_mode: str = "default"


    # Overlay settings
    font_family: str = "Malgun Gothic"
    font_size: int = 25

    # Vertical space for comments.
    overlay_top_ratio: float = 0.03
    overlay_bottom_ratio: float = 0.30

    # Lane settings.
    lane_height_px: int = 30
    lane_vertical_padding_px: int = 10
    min_comment_gap_px: int = 140

    max_simultaneous_comments: int = 15
    max_pending_comments: int = 60

    animation_interval_ms: int = 33
    comment_spawn_min_interval_ms: int = 1500
    comment_spawn_max_interval_ms: int = 4000
    comment_speed_px_per_tick: float = 18.0

    clear_active_comments_on_new_batch: bool = False


@dataclass(slots=True)
class CaptureFrame:
    """A captured screen frame, plus optional OCR text."""

    image_path: Path
    timestamp: float
    ocr_text: str | None = None


@dataclass(slots=True)
class CommentBatch:
    """Structured comments returned by the API module."""

    comments: list[str] = field(default_factory=list)
    long_comments: list[str] = field(default_factory=list)

    # S: rewritten whole/canonical summary.
    summary: str = ""

    # T: short factual snapshot of the current capture only.
    current_situation: str = ""

    is_error: bool = False
    error_message: str = ""

    @classmethod
    def error(cls, message: str) -> "CommentBatch":
        return cls(
            comments=[],
            long_comments=[],
            summary="",
            current_situation="",
            is_error=True,
            error_message=message,
        )

    @classmethod
    def dummy(cls) -> "CommentBatch":
        return cls(
            comments=[
                "ㅋㅋㅋㅋㅋ",
                "ㅋㅋ",
                "오",
            ],
            long_comments=[
                "재미있어요",
            ],
            summary="Dummy response: characters appear to be continuing a scene.",
            is_error=False,
            error_message="",
        )
