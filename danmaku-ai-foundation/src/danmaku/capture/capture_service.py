from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from danmaku.models import CaptureFrame


class CaptureService:
    """
    Basic screen/window capture service.

    If target_window_title is empty, captures the full primary screen.
    If target_window_title is set, captures that visible window rectangle.
    """

    def __init__(
        self,
        output_dir: Path,
        target_window_title: str = "",
        image_format: str = "PNG",
        jpeg_quality: int = 82,
    ) -> None:
        self.output_dir = output_dir
        self.target_window_title = target_window_title
        self.image_format = image_format.upper()
        self.jpeg_quality = max(20, min(95, int(jpeg_quality)))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def set_target_window_title(self, title: str) -> None:
        self.target_window_title = title.strip()

    def capture(self) -> CaptureFrame:
        image_path = self._make_capture_path()

        if self.target_window_title:
            self._capture_window(image_path, self.target_window_title)
        else:
            self._capture_full_screen(image_path)

        return CaptureFrame(
            image_path=image_path,
            timestamp=time.time(),
            ocr_text=None,
        )

    def _make_capture_path(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        safe_target = "fullscreen"

        if self.target_window_title:
            safe_target = "".join(
                char if char.isalnum() or char in ("-", "_") else "_"
                for char in self.target_window_title[:40]
            )

        suffix = ".jpg" if self.image_format == "JPEG" else ".png"
        return self.output_dir / f"capture_{safe_target}_{timestamp}{suffix}"

    def _save_image(self, image, output_path: Path) -> None:
        if self.image_format == "JPEG":
            image.convert("RGB").save(
                output_path,
                format="JPEG",
                quality=self.jpeg_quality,
                optimize=True,
            )
            return

        image.save(output_path, format="PNG")

    def _capture_full_screen(self, output_path: Path) -> None:
        try:
            import mss
            from PIL import Image

            with mss.mss() as sct:
                monitor = sct.monitors[1]
                screenshot = sct.grab(monitor)
                image = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
                self._save_image(image, output_path)
                return
        except Exception:
            pass

        try:
            from PIL import ImageGrab

            image = ImageGrab.grab()
            self._save_image(image, output_path)
            return
        except Exception as exc:
            raise RuntimeError(f"Full-screen capture failed: {exc}") from exc

    def _capture_window(self, output_path: Path, window_title: str) -> None:
        """
        Captures the visible screen rectangle occupied by the selected window.

        Limitation:
        - The window must be visible.
        - If another window covers it, the covered pixels may also be captured.
        - Minimized windows cannot be captured this way.
        """
        try:
            import pygetwindow as gw
            import mss
            from PIL import Image

            matches = [
                window
                for window in gw.getWindowsWithTitle(window_title)
                if window.title.strip()
            ]

            if not matches:
                raise RuntimeError(
                    f"No window found with title containing: {window_title}")

            window = matches[0]

            if window.isMinimized:
                raise RuntimeError(
                    f"Selected window is minimized: {window.title}")

            left = int(window.left)
            top = int(window.top)
            width = int(window.width)
            height = int(window.height)

            if width <= 0 or height <= 0:
                raise RuntimeError(
                    f"Selected window has invalid size: {window.title}")

            region = {
                "left": left,
                "top": top,
                "width": width,
                "height": height,
            }

            with mss.mss() as sct:
                screenshot = sct.grab(region)
                image = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
                self._save_image(image, output_path)

            return

        except Exception as exc:
            raise RuntimeError(f"Window capture failed: {exc}") from exc


def list_window_titles() -> list[str]:
    import pygetwindow as gw

    titles = sorted(
        {
            window.title.strip()
            for window in gw.getAllWindows()
            if window.title and window.title.strip()
        }
    )

    return titles


def main() -> None:
    print("Available windows:")
    for title in list_window_titles():
        print("-", title)

    service = CaptureService(output_dir=Path("logs/captures"))
    frame = service.capture()
    print(f"Captured: {frame.image_path.resolve()}")


if __name__ == "__main__":
    main()
