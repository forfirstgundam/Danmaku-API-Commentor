from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from danmaku.models import CaptureFrame


class CaptureService:
    """
    Basic screen/window/region capture service.

    Modes:
    - full_screen: captures the primary screen.
    - window: captures a specific window rectangle.
    - region: captures a user-selected region.
    """

    def __init__(
        self,
        output_dir: Path,
        capture_mode: str = "full_screen",
        capture_region: tuple[int, int, int, int] = (0, 0, 0, 0),
        target_window_title: str = "",
    ) -> None:
        self.output_dir = output_dir
        self.capture_mode = capture_mode
        self.capture_region = capture_region
        self.target_window_title = target_window_title
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def set_target_window_title(self, title: str) -> None:
        self.target_window_title = title.strip()

    def capture(self) -> CaptureFrame:
        image_path = self._make_capture_path()

        if self.capture_mode == "region":
            self._capture_region(image_path, self.capture_region)
        elif self.capture_mode == "window" and self.target_window_title:
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
        if self.capture_mode == "region":
            safe_target = "region"
        elif self.target_window_title:
            safe_target = "".join(
                char if char.isalnum() or char in ("-", "_") else "_"
                for char in self.target_window_title[:40]
            )
        else:
            safe_target = "fullscreen"

        return self.output_dir / f"capture_{safe_target}_{timestamp}.png"

    def _capture_full_screen(self, output_path: Path) -> None:
        try:
            import mss
            from PIL import Image

            with mss.mss() as sct:
                monitor = sct.monitors[1]
                screenshot = sct.grab(monitor)
                image = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
                image.save(output_path)
                return
        except Exception:
            pass

        try:
            from PIL import ImageGrab

            image = ImageGrab.grab()
            image.save(output_path)
            return
        except Exception as exc:
            raise RuntimeError(f"Full-screen capture failed: {exc}") from exc

    def _capture_region(self, output_path: Path, region: tuple[int, int, int, int]) -> None:
        try:
            import mss
            from PIL import Image

            left, top, width, height = region
            if width <= 0 or height <= 0:
                raise RuntimeError("Invalid capture region size")

            region_box = {
                "left": left,
                "top": top,
                "width": width,
                "height": height,
            }

            with mss.mss() as sct:
                screenshot = sct.grab(region_box)
                image = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
                image.save(output_path)
                return
        except Exception as exc:
            raise RuntimeError(f"Region capture failed: {exc}") from exc

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
                image.save(output_path)

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
