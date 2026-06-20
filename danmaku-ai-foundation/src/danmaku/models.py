from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class AppSettings:
    """Runtime settings shared across modules."""
    first_capture_delay_ms: int = 1500

    capture_interval_seconds: int = 6
    model_name: str = "gemini-2.5-flash-lite"
    api_key: str = ""
    use_dummy_api: bool = True

    # Testing / logging
    save_captures: bool = True
    save_comments: bool = True

    # Base folder for all runs
    log_root_dir: Path = Path("logs")

    # These are filled/replaced when Start is pressed
    run_log_dir: Path = Path("logs/current")
    capture_output_dir: Path = Path("logs/current/captures")
    comment_log_path: Path = Path("logs/current/comments.jsonl")

    # Capture settings
    target_window_title: str = ""  # empty means full screen
    capture_mode: str = "full_screen"
    capture_region: tuple[int, int, int, int] = (0, 0, 0, 0)

    # Overlay settings
    font_family: str = "Malgun Gothic"
    font_size: int = 20

    # Vertical space for comments.
    overlay_top_ratio: float = 0.03
    overlay_bottom_ratio: float = 0.36

    # Lane settings.
    lane_height_px: int = 25
    lane_vertical_padding_px: int = 5
    min_comment_gap_px: int = 160

    max_simultaneous_comments: int = 15
    max_pending_comments: int = 60

    animation_interval_ms: int = 33
    comment_spawn_min_interval_ms: int = 2000
    comment_spawn_max_interval_ms: int = 4000
    comment_speed_px_per_tick: float = 12.0

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
