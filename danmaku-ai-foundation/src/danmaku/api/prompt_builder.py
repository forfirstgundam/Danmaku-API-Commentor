from __future__ import annotations

from danmaku.config import load_text_file, resource_path
from danmaku.models import CaptureFrame


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
    ) -> str:
        ocr_text = frame.ocr_text or ""
        recent_comment_text = "\n".join(
            f"- {comment}"
            for comment in (previous_comments or [])
            if comment.strip()
        )

        return f"""
Analyze the current screenshot and generate Korean danmaku-style reaction comments.

You are given five kinds of information:

1. Previous whole summary S(n-1):
A rewritten factual summary of the whole story or viewing session from the
beginning, including important earlier chapters and major events.

2. Recent current-situation snapshots T(n-k):
Factual descriptions of the most recent individual screenshots,
ordered from oldest to newest.

3. Current OCR text:
Text extracted from the dialogue or subtitle area.
It may be incomplete or contain recognition errors.

4. Current screenshot:
The current visual state. This is the most important source of truth.

5. Recent generated comments:
Comments that were already generated recently.
Use these only to avoid repetition.

Previous context:
{previous_summary or "(none)"}

Current OCR text:
{ocr_text or "(none)"}

Recent generated comments:
{recent_comment_text or "(none)"}

Return strict JSON with this schema:
{{
  "comments": ["short comment", "short comment"],
  "long_comments": ["longer reaction comment"],
  "summary": "S(n): rewritten whole factual summary after incorporating the current screenshot",
  "current_situation": "T(n): factual snapshot of only the current screenshot/current moment"
}}

Comment requirements:
- Write comments in Korean.
- Use Korean internet stream chat / danmaku style.
- Use casual slang, meme-like reactions, and short viewer comments.
- Avoid formal explanatory comments.
- React to the overall situation, not only isolated visible objects.
- Do not pretend to know the real player's thoughts or feelings.
- Do not repeat recent generated comments.
- Avoid near-duplicates that only change particles, punctuation, wording, or laughter.
- Generate 3 to 6 short comments.
- Generate 0 to 1 longer comment.

Source-priority requirements:
- The current screenshot is the most important source of truth.
- If previous context conflicts with the current screenshot, trust the current screenshot.
- Use previous S and recent T snapshots to understand cut-off text,
  partial dialogue, and scene continuity.
- OCR may be incomplete or incorrect; confirm it against the screenshot.
- Do not invent dialogue, names, or events unsupported by the available information.
- Do not overreact to one partial screenshot if recent T snapshots clarify it.

Scene-change requirements:
- Use "[SCENE_CHANGE]" only as a full context-reset signal: the viewer has
  clearly switched to a different, unrelated game, video, work, or activity.
- Do not use "[SCENE_CHANGE]" for an ordinary camera cut, location change,
  menu, page, chapter transition, new character, or new scene within the same
  continuing story.
- For normal transitions within the same work, preserve S(n-1) and incorporate
  the new event into the whole story summary.
- For a genuine unrelated-content reset, ignore the old story and start
  "summary" with "[SCENE_CHANGE] ".

Summary S(n) requirements:
- "summary" is the rewritten whole/canonical factual summary.
- It replaces the previous whole summary S(n-1).
- Rewrite the whole summary after incorporating the current screenshot.
- Do not merely append the current situation to the previous summary.
- Do not copy recent T snapshots as a list.
- Preserve important story progression from the beginning, including prior
  chapters and major events, even when the visual scene changes.
- Compress minor or repetitive details, but do not reduce S to only the latest
  screenshot or recent scene.
- Preserve important continuity such as the game or video if known,
  characters, relationships, locations, completed major events, dialogue or
  event, menu state, selected choices, and recent story situation.
- Remove only contradicted information and details too minor to help understand
  the continuing story.
- Keep the summary factual and objective.
- Include only facts directly visible, explicitly stated in dialogue/text, or
  already established in S(n-1).
- Describe observable evidence instead of interpretation: use "the character
  frowns" rather than "the mood becomes confrontational."
- Do not infer mood, tone, motives, relationships, future events, danger,
  foreshadowing, or what an object might imply.
- Do not include jokes, slang, audience reactions, or player emotions.
- Keep it concise, but use enough sentences to retain major chapter-level
  progression from the beginning.

Current situation T(n) requirements:
- "current_situation" describes only the current screenshot and moment.
- It must be factual and objective.
- Use exactly 1 concise sentence.
- It may mention visible dialogue, menu state, characters, actions,
  loading state, or error state.
- Include only visible or explicitly written information.
- Do not infer mood, tone, motives, implications, or future events.
- Do not include audience reactions, jokes, slang, or player emotions.
- Do not summarize the entire story here; that belongs in "summary".

Do not include Markdown.
Do not include explanations outside JSON.
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
