from __future__ import annotations

import json
import re

from danmaku.api.prompt_builder import PromptBuilder
from danmaku.models import CaptureFrame, CommentBatch


class GeminiLLMClient:
    """
    Gemini API client.

    By default, this can run in dummy mode so the overlay and capture modules
    can be developed without spending API calls.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash-lite",
        use_dummy_api: bool = True,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.use_dummy_api = use_dummy_api
        self.prompt_builder = prompt_builder or PromptBuilder()

    def generate_comments(self, frame: CaptureFrame, previous_summary: str = "") -> CommentBatch:
        if self.use_dummy_api or not self.api_key:
            return self._dummy_response()

        try:
            return self._generate_with_gemini(frame, previous_summary)
        except Exception as exc:
            message = f"Gemini call failed: {exc}"
            print(f"[api] {message}")
            return CommentBatch.error(message)

    def _generate_with_gemini(self, frame: CaptureFrame, previous_summary: str) -> CommentBatch:
        from google import genai
        from google.genai import types

        image_bytes = frame.image_path.read_bytes()
        system_prompt = self.prompt_builder.build_system_prompt()
        user_prompt = self.prompt_builder.build_user_prompt(
            frame, previous_summary)

        client = genai.Client(api_key=self.api_key)

        response = client.models.generate_content(
            model=self.model_name,
            contents=[
                user_prompt,
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
            ),
        )

        return self._parse_comment_batch(response.text or "")

    def _parse_comment_batch(self, text: str) -> CommentBatch:
        cleaned = self._strip_code_fence(text).strip()
        data = json.loads(cleaned)

        comments = data.get("comments", [])
        long_comments = data.get("long_comments", [])
        summary = data.get("summary", "")
        current_situation = data.get("current_situation", "")

        if not isinstance(comments, list):
            comments = []
        if not isinstance(long_comments, list):
            long_comments = []
        if not isinstance(summary, str):
            summary = ""
        if not isinstance(current_situation, str):
            current_situation = ""

        comments = [str(item) for item in comments if str(item).strip()]
        long_comments = [str(item)
                         for item in long_comments if str(item).strip()]

        if not comments and not long_comments:
            return CommentBatch.error("Gemini returned no comments.")

        return CommentBatch(
            comments=comments[:12],
            long_comments=long_comments[:3],
            summary=summary.strip(),
            current_situation=current_situation.strip(),
        )

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        pattern = r"^```(?:json)?\s*(.*?)\s*```$"
        match = re.match(pattern, text.strip(), re.DOTALL | re.IGNORECASE)
        return match.group(1) if match else text

    @staticmethod
    def _dummy_response() -> CommentBatch:
        return CommentBatch.dummy()


def main() -> None:
    from pathlib import Path
    from danmaku.models import CaptureFrame

    client = GeminiLLMClient(api_key="", use_dummy_api=True)
    frame = CaptureFrame(image_path=Path("example.png"), timestamp=0)
    batch = client.generate_comments(frame)
    print(batch)


if __name__ == "__main__":
    main()
