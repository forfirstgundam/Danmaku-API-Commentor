from __future__ import annotations

import base64
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Callable

from danmaku.api.prompt_builder import PromptBuilder
from danmaku.models import CaptureFrame, CommentBatch


class LLMClient:
    """
    Gemini API client.

    By default, this can run in dummy mode so the overlay and capture modules
    can be developed without spending API calls.
    """

    def __init__(
        self,
        api_key: str,
        api_provider: str = "gemini",
        model_name: str = "gemini-2.5-flash-lite",
        use_dummy_api: bool = True,
        send_screenshot: bool = True,
        image_max_dimension: int = 768,
        image_jpeg_quality: int = 72,
        max_output_tokens: int = 512,
        save_api_images: bool = False,
        api_image_output_dir: Path | None = None,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        self.api_key = api_key
        self.api_provider = api_provider.lower()
        self.model_name = model_name
        self.use_dummy_api = use_dummy_api
        self.send_screenshot = send_screenshot
        self.image_max_dimension = image_max_dimension
        self.image_jpeg_quality = image_jpeg_quality
        self.max_output_tokens = max_output_tokens
        self.save_api_images = save_api_images
        self.api_image_output_dir = api_image_output_dir
        self.prompt_builder = prompt_builder or PromptBuilder()
        self._openai_client = None

        if self.api_provider == "openai" and self.api_key and not self.use_dummy_api:
            from openai import OpenAI

            self._openai_client = OpenAI(api_key=self.api_key)

    def generate_comments(
        self,
        frame: CaptureFrame,
        previous_summary: str = "",
        previous_comments: list[str] | None = None,
        use_streaming: bool = False,
        on_comment: Callable[[str], None] | None = None,
    ) -> CommentBatch:
        if self.use_dummy_api or not self.api_key:
            return self._dummy_response()

        try:
            if self.api_provider == "openai":
                return self._generate_with_openai(
                    frame,
                    previous_summary,
                    previous_comments or [],
                )
            if use_streaming and on_comment is not None:
                return self._generate_with_gemini_stream(
                    frame,
                    previous_summary,
                    previous_comments or [],
                    on_comment,
                )
            return self._generate_with_gemini(
                frame,
                previous_summary,
                previous_comments or [],
            )
        except Exception as exc:
            message = f"{self.api_provider.title()} call failed: {exc}"
            print(f"[api] {message}")
            return CommentBatch.error(message)

    def _generate_with_gemini(
        self,
        frame: CaptureFrame,
        previous_summary: str,
        previous_comments: list[str],
    ) -> CommentBatch:
        from google import genai
        from google.genai import types

        system_prompt = self.prompt_builder.build_system_prompt()
        user_prompt = self.prompt_builder.build_user_prompt(
            frame,
            previous_summary,
            previous_comments,
        )

        client = genai.Client(api_key=self.api_key)

        contents: list[object] = [user_prompt]

        if self.send_screenshot:
            image_bytes, mime_type = self._build_api_image(frame)
            contents.append(
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            )
        else:
            print("[api] text-only request: screenshot not sent")

        thinking_config = None
        if self.model_name.startswith("gemini-2.5-flash"):
            thinking_config = types.ThinkingConfig(thinking_budget=0)
        elif self.model_name == "gemini-2.5-pro":
            thinking_config = types.ThinkingConfig(thinking_budget=128)
        elif self.model_name.startswith("gemini-3"):
            thinking_config = types.ThinkingConfig(thinking_level="minimal")

        response = client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                max_output_tokens=self.max_output_tokens,
                thinking_config=thinking_config,
            ),
        )

        return self._parse_comment_batch(response.text or "")

    def _generate_with_gemini_stream(
        self,
        frame: CaptureFrame,
        previous_summary: str,
        previous_comments: list[str],
        on_comment: Callable[[str], None],
    ) -> CommentBatch:
        from google import genai
        from google.genai import types

        system_prompt = self.prompt_builder.build_system_prompt()
        user_prompt = self.prompt_builder.build_user_prompt(
            frame,
            previous_summary,
            previous_comments,
        )

        client = genai.Client(api_key=self.api_key)
        contents: list[object] = [user_prompt]

        if self.send_screenshot:
            image_bytes, mime_type = self._build_api_image(frame)
            contents.append(
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            )
        else:
            print("[api] text-only request: screenshot not sent")

        thinking_config = None
        if self.model_name.startswith("gemini-2.5-flash"):
            thinking_config = types.ThinkingConfig(thinking_budget=0)
        elif self.model_name == "gemini-2.5-pro":
            thinking_config = types.ThinkingConfig(thinking_budget=128)
        elif self.model_name.startswith("gemini-3"):
            thinking_config = types.ThinkingConfig(thinking_level="minimal")

        chunks: list[str] = []
        emitted_count = 0

        stream = client.models.generate_content_stream(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                max_output_tokens=self.max_output_tokens,
                thinking_config=thinking_config,
            ),
        )

        for chunk in stream:
            text = chunk.text or ""
            if not text:
                continue

            chunks.append(text)
            full_text = "".join(chunks)
            comments = self._extract_partial_comments(full_text)

            for comment in comments[emitted_count:]:
                on_comment(comment)
                emitted_count += 1

        return self._parse_comment_batch("".join(chunks))

    def _generate_with_openai(
        self,
        frame: CaptureFrame,
        previous_summary: str,
        previous_comments: list[str],
    ) -> CommentBatch:
        system_prompt = self.prompt_builder.build_system_prompt()
        user_prompt = self.prompt_builder.build_user_prompt(
            frame,
            previous_summary,
            previous_comments,
        )
        content: list[dict[str, object]] = [
            {"type": "input_text", "text": user_prompt}
        ]

        if self.send_screenshot:
            image_bytes, mime_type = self._build_api_image(frame)
            encoded = base64.b64encode(image_bytes).decode("ascii")
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{mime_type};base64,{encoded}",
                    "detail": "low",
                }
            )
        else:
            print("[api] text-only request: screenshot not sent")

        if self._openai_client is None:
            raise RuntimeError("OpenAI client is not initialized.")

        reasoning_effort = "minimal" if self.model_name in {
            "gpt-5-mini",
            "gpt-5-nano",
        } else "none"

        response = self._openai_client.responses.create(
            model=self.model_name,
            instructions=system_prompt,
            input=[{"role": "user", "content": content}],
            reasoning={"effort": reasoning_effort},
            max_output_tokens=self.max_output_tokens,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "comment_batch",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "comments": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "long_comments": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "summary": {"type": "string"},
                        },
                        "required": ["comments", "long_comments", "summary"],
                        "additionalProperties": False,
                    },
                }
            },
        )

        return self._parse_comment_batch(response.output_text or "")

    def _build_api_image(self, frame: CaptureFrame) -> tuple[bytes, str]:
        if self.image_max_dimension <= 0:
            return frame.image_path.read_bytes(), "image/png"

        from PIL import Image

        with Image.open(frame.image_path) as image:
            original_size = image.size
            image = image.convert("RGB")
            image.thumbnail(
                (self.image_max_dimension, self.image_max_dimension),
                Image.Resampling.LANCZOS,
            )

            buffer = BytesIO()
            jpeg_quality = max(20, min(95, int(self.image_jpeg_quality)))
            image.save(buffer, format="JPEG",
                       quality=jpeg_quality, optimize=True)
            image_bytes = buffer.getvalue()

        if self.save_api_images and self.api_image_output_dir:
            self.api_image_output_dir.mkdir(parents=True, exist_ok=True)
            safe_model_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.model_name)
            api_image_path = self.api_image_output_dir / (
                f"{frame.image_path.stem}_{safe_model_name}_api_"
                f"{image.size[0]}x{image.size[1]}.jpg"
            )
            api_image_path.write_bytes(image_bytes)
            print(f"[api] saved request image: {api_image_path}")

        print(
            "[api] resized image for request: "
            f"{original_size[0]}x{original_size[1]} -> "
            f"{image.size[0]}x{image.size[1]} "
            f"quality={jpeg_quality} "
            f"({round(len(image_bytes) / 1024, 1)} KB)"
        )
        return image_bytes, "image/jpeg"

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
            return CommentBatch.error(
                f"{self.api_provider.title()} returned no comments.")

        return CommentBatch(
            comments=comments[:12],
            long_comments=long_comments[:3],
            summary=summary.strip(),
            current_situation=current_situation.strip(),
        )

    @staticmethod
    def _extract_partial_comments(text: str) -> list[str]:
        match = re.search(r'"comments"\s*:\s*\[', text)

        if not match:
            return []

        decoder = json.JSONDecoder()
        comments: list[str] = []
        index = match.end()

        while index < len(text):
            while index < len(text) and text[index] in " \r\n\t,":
                index += 1

            if index >= len(text) or text[index] == "]":
                break

            if text[index] != '"':
                index += 1
                continue

            try:
                value, next_index = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                break

            if isinstance(value, str) and value.strip():
                comments.append(value.strip())

            index += next_index

        return comments

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

    client = LLMClient(api_key="", use_dummy_api=True)
    frame = CaptureFrame(image_path=Path("example.png"), timestamp=0)
    batch = client.generate_comments(frame)
    print(batch)


if __name__ == "__main__":
    main()


GeminiLLMClient = LLMClient
