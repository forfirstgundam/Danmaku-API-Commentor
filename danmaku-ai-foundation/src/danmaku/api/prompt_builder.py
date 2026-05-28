from __future__ import annotations

from danmaku.config import load_text_file, resource_path
from danmaku.models import CaptureFrame


class PromptBuilder:
    """Builds prompts for danmaku comment generation."""

    def __init__(self, system_prompt_path: str = "prompts/system_prompt.txt") -> None:
        self.system_prompt_path = system_prompt_path

    def build_system_prompt(self) -> str:
        default_prompt = (
            "You generate short Korean danmaku/tvple-style reaction comments from a game "
            "or video screenshot. Return strict JSON only."
        )
        return load_text_file(resource_path(self.system_prompt_path), default_prompt)

    def build_user_prompt(self, frame: CaptureFrame, previous_summary: str) -> str:
        ocr_text = frame.ocr_text or ""

        return f"""
        Analyze the current screenshot and generate Korean danmaku-style reaction comments.

        You are given:
        1. Previous whole summary S(n-1)
        2. Recent current-situation snapshots T(n-k)
        3. Current OCR text
        4. Current screenshot

        Context:
        {previous_summary or "(none)"}

        Current OCR text:
        {ocr_text or "(none)"}

        Return strict JSON with this schema:
        {{
        "comments": ["short comment", "short comment"],
        "long_comments": ["longer reaction comment"],
        "summary": "S(n): rewritten whole factual summary after incorporating the current screenshot",
        "current_situation": "T(n): factual snapshot of only the current screenshot/current moment"
        }}

        Comment requirements:
        - Write comments in Korean.
        - Style should feel like Korean internet stream chat / danmaku.
        - Use casual slang, meme-like reactions, and short chaotic viewer comments.
        - Avoid formal explanatory comments.
        - Avoid comments that only describe visible objects.
        - Do not pretend to know the real player's feelings.
        - Generate 6 short comments.
        - Generate 2 longer comments.

        Summary S(n) requirements:
        - "summary" is the rewritten whole/canonical factual summary.
        - It replaces the previous whole summary S(n-1).
        - Rewrite the whole summary every time after incorporating the current screenshot.
        - Do not merely append the current scene to the previous summary.
        - Do not copy the recent T snapshots as a list.
        - Keep only information useful for understanding future screenshots.
        - Preserve important continuity: game title if known, characters, location, current dialogue/event, menu state, selected choices, and recent story situation.
        - Remove outdated details that are no longer useful.
        - Summary must be factual and objective.
        - Do not include jokes, slang, audience reactions, or player emotions in the summary.
        - Summary should usually be 2 to 4 concise sentences.

        Current situation T(n) requirements:
        - "current_situation" describes only the current screenshot/current moment.
        - It should be factual and objective.
        - It should be 1~3 concise sentence.
        - It may mention visible dialogue, menu state, character/action, or loading/error state.
        - Do not include audience reactions, jokes, slang, or player emotions.
        - Do not summarize the whole story here; that belongs in "summary".

        Context-use requirements:
        - Use previous S and recent T snapshots to understand cut-off text or partial dialogue.
        - If the current screenshot only shows part of a sentence, infer continuity from S and recent T, but mark uncertainty when needed.
        - Do not overreact to a single partial screenshot if recent T snapshots clarify the situation.

        Do not include Markdown.
        Do not include explanations outside JSON.
        """.strip()


def main() -> None:
    from pathlib import Path
    from danmaku.models import CaptureFrame

    builder = PromptBuilder()
    frame = CaptureFrame(image_path=Path("example.png"),
                         timestamp=0, ocr_text="こんにちは")
    print("SYSTEM PROMPT:")
    print(builder.build_system_prompt())
    print("\nUSER PROMPT:")
    print(builder.build_user_prompt(
        frame, previous_summary="A character entered the scene."))


if __name__ == "__main__":
    main()
