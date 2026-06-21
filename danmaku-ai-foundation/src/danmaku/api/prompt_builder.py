from __future__ import annotations

from danmaku.config import load_text_file, resource_path
from danmaku.models import CaptureFrame, SessionProfile


class PromptBuilder:
    """Builds prompts for danmaku comment generation."""

    def __init__(
        self,
        system_prompt_path: str = "prompts/system_prompt.txt",
    ) -> None:
        self.system_prompt_path = system_prompt_path

    def build_system_prompt(self) -> str:
        default_prompt = (
            "You generate short Korean danmaku/tvple-style reaction comments "
            "from a game or video screenshot. Return strict JSON only."
        )
        return load_text_file(
            resource_path(self.system_prompt_path),
            default_prompt,
        )

    def build_user_prompt(
        self,
        frame: CaptureFrame,
        previous_summary: str,
        previous_comments: list[str] | None = None,
        context_frames: list[CaptureFrame] | None = None,
        user_stream_description: str = "",
        session_profile: SessionProfile | None = None,
    ) -> str:
        ocr_text = frame.ocr_text or ""
        frames = [*(context_frames or []), frame]
        current_timestamp = frame.timestamp
        frame_timeline = "\n".join(
            (
                f"- Frame {index + 1}: "
                f"{max(0.0, current_timestamp - item.timestamp):.1f} seconds "
                f"before the latest frame"
                + (" (LATEST/CURRENT)" if item is frame else "")
            )
            for index, item in enumerate(frames)
        )
        recent_comment_text = "\n".join(
            f"- {comment}"
            for comment in (previous_comments or [])
            if comment.strip()
        )
        profile_text = (
            session_profile.to_prompt_text()
            if session_profile is not None
            else "- Title: unknown\n- Content type: unknown"
        )

        return f"""
Generate the next danmaku batch using the system rules and the request data
below.

## Session context

User description:
{user_stream_description.strip() or "(none)"}

Interpreted profile:
{profile_text}

## Memory before this request

{previous_summary or "(none)"}

## Current OCR

{ocr_text or "(none)"}

## Recent generated comments

{recent_comment_text or "(none)"}

## Attached frame timeline

{frame_timeline}

The attached images follow this order. Frame {len(frames)} is the fresh
latest/current frame. Produce the required JSON response now.
""".strip()


def main() -> None:
    from pathlib import Path

    builder = PromptBuilder()
    frame = CaptureFrame(
        image_path=Path("example.png"),
        timestamp=0,
        ocr_text="こんにちは",
    )

    print("SYSTEM PROMPT:")
    print(builder.build_system_prompt())
    print("\nUSER PROMPT:")
    print(
        builder.build_user_prompt(
            frame,
            previous_summary="A character entered the scene.",
            previous_comments=["ㅋㅋㅋ 뭐야", "이건 좀 웃기네"],
        )
    )


if __name__ == "__main__":
    main()
