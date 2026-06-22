from __future__ import annotations

import threading
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


class OcrUnavailableError(RuntimeError):
    """Raised when the optional local OCR runtime is unavailable."""


@dataclass(slots=True)
class OcrObservation:
    timestamp: float
    text: str
    confidence: float

    def to_dict(self, latest_timestamp: float) -> dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "age_from_latest_sec": round(
                max(0.0, latest_timestamp - self.timestamp),
                3,
            ),
            "text": self.text,
            "confidence": round(self.confidence, 3),
        }


class EasyOcrService:
    """Lazy, reusable EasyOCR reader for one configured language set."""

    def __init__(self, languages: tuple[str, ...]) -> None:
        self.languages = tuple(dict.fromkeys(languages)) or ("en",)
        self._reader = None
        self._initialization_error = ""
        self._lock = threading.Lock()

    def _get_reader(self):
        if self._reader is not None:
            return self._reader
        if self._initialization_error:
            raise OcrUnavailableError(self._initialization_error)

        try:
            import easyocr
            # CPU is the predictable default. EasyOCR may download its
            # language model on first use, then reuses it for the run.
            self._reader = easyocr.Reader(list(self.languages), gpu=False)
        except Exception as exc:
            self._initialization_error = (
                "EasyOCR could not initialize and has been disabled for this "
                f"run: {exc}"
            )
            raise OcrUnavailableError(self._initialization_error) from exc
        return self._reader

    def prepare(self) -> None:
        """Download/load the configured model before a stream starts."""
        with self._lock:
            self._get_reader()

    def recognize(
        self,
        image_path: Path,
        region: tuple[float, float, float, float],
        min_confidence: float,
    ) -> tuple[str, float]:
        try:
            from PIL import Image
        except ImportError as exc:
            raise OcrUnavailableError(
                "OCR dependencies are unavailable. Run: pip install -r requirements.txt"
            ) from exc

        with Image.open(image_path) as source:
            image = source.convert("RGB")
            crop = crop_normalized_region(image, region)

        return self.recognize_image(crop, min_confidence)

    def recognize_image(
        self,
        image,
        min_confidence: float,
    ) -> tuple[str, float]:
        """Recognize an already-cropped in-memory PIL image."""
        try:
            import numpy as np
            from PIL import Image
        except ImportError as exc:
            raise OcrUnavailableError(
                "OCR dependencies are unavailable. Run: pip install -r requirements.txt"
            ) from exc

        crop = image.convert("RGB")
        # OCR detection cost grows sharply with large captures. The API
        # screenshots can remain high-resolution; only the OCR crop needs
        # this bounded working size.
        if max(crop.size) > 1280:
            crop.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
        pixels = np.asarray(crop)

        with self._lock:
            results = self._get_reader().readtext(
                pixels,
                detail=1,
                paragraph=False,
            )

        accepted: list[tuple[float, float, str, float]] = []
        for result in results:
            if not isinstance(result, (list, tuple)) or len(result) < 3:
                continue
            box, text, confidence = result[0], result[1], result[2]
            clean = clean_ocr_text(str(text))
            try:
                score = float(confidence)
            except (TypeError, ValueError):
                score = 0.0
            if not clean or score < min_confidence:
                continue
            try:
                x = min(float(point[0]) for point in box)
                y = min(float(point[1]) for point in box)
            except (TypeError, ValueError, IndexError):
                x, y = 0.0, float(len(accepted))
            accepted.append((y, x, clean, score))

        accepted.sort(key=lambda item: (round(item[0] / 12), item[1]))
        text = " ".join(item[2] for item in accepted)
        confidence = (
            sum(item[3] for item in accepted) / len(accepted)
            if accepted
            else 0.0
        )
        return clean_ocr_text(text), confidence


class RollingOcrBuffer:
    """Thread-safe, short-lived OCR timeline between API requests."""

    def __init__(self, max_items: int = 12) -> None:
        self.max_items = max(1, max_items)
        self._items: list[OcrObservation] = []
        self._lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def add(self, observation: OcrObservation) -> bool:
        clean = clean_ocr_text(observation.text)
        if not clean:
            return False
        observation.text = clean
        normalized = normalize_for_comparison(clean)

        with self._lock:
            if self._items:
                previous = self._items[-1]
                previous_normalized = normalize_for_comparison(previous.text)
                similarity = SequenceMatcher(
                    None,
                    previous_normalized,
                    normalized,
                ).ratio()
                if normalized == previous_normalized or similarity >= 0.96:
                    # Keep the newest timestamp/confidence without repeating
                    # a subtitle that remained on screen for several samples.
                    self._items[-1] = observation
                    return False

            self._items.append(observation)
            self._items = self._items[-self.max_items:]
            return True

    def drain(
        self,
        latest_timestamp: float,
    ) -> tuple[str, list[dict[str, object]]]:
        with self._lock:
            items = list(self._items)
            self._items.clear()

        records = [item.to_dict(latest_timestamp) for item in items]
        if not items:
            return "", records

        lines = [
            f"- {max(0.0, latest_timestamp - item.timestamp):.1f}s ago: "
            f"{item.text}"
            for item in items
        ]
        return (
            "Recent OCR observations, oldest to newest:\n"
            + "\n".join(lines),
            records,
        )


def normalized_region_to_box(
    region: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x, y, width, height = region
    x = min(1.0, max(0.0, float(x)))
    y = min(1.0, max(0.0, float(y)))
    width = min(1.0 - x, max(0.001, float(width)))
    height = min(1.0 - y, max(0.001, float(height)))
    left = min(image_width - 1, max(0, round(x * image_width)))
    top = min(image_height - 1, max(0, round(y * image_height)))
    right = min(image_width, max(left + 1, round((x + width) * image_width)))
    bottom = min(
        image_height,
        max(top + 1, round((y + height) * image_height)),
    )
    return left, top, right, bottom


def crop_normalized_region(
    image,
    region: tuple[float, float, float, float],
):
    left, top, right, bottom = normalized_region_to_box(
        region,
        image.width,
        image.height,
    )
    return image.crop((left, top, right, bottom))


def make_visual_signature(image, width: int = 64, height: int = 32) -> bytes:
    """Create a tiny grayscale signature for inexpensive change detection."""
    from PIL import Image

    return image.convert("L").resize(
        (width, height),
        Image.Resampling.BILINEAR,
    ).tobytes()


def visual_signature_difference(previous: bytes, current: bytes) -> float:
    if not previous or len(previous) != len(current):
        return 1.0
    return sum(
        abs(before - after)
        for before, after in zip(previous, current)
    ) / (len(current) * 255.0)


def clean_ocr_text(text: str) -> str:
    return " ".join(text.replace("\x00", " ").split()).strip()


def normalize_for_comparison(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold().replace(" ", "")
