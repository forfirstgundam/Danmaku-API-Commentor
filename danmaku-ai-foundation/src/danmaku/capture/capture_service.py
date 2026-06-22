from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from danmaku.models import CaptureFrame


class CaptureService:
    """
    Basic screen/window capture service.

    If target_window_title is empty, captures the full primary screen.
    A selected window is captured from its stable Windows handle, so title
    changes and overlapping windows do not affect the captured pixels.
    """

    def __init__(
        self,
        output_dir: Path,
        capture_mode: str = "full_screen",
        capture_region: tuple[int, int, int, int] = (0, 0, 0, 0),
        target_window_title: str = "",
        target_window_handle: int = 0,
        image_format: str = "PNG",
        jpeg_quality: int = 82,
    ) -> None:
        self.output_dir = output_dir
        self.capture_mode = capture_mode
        self.capture_region = capture_region
        self.target_window_title = target_window_title
        self.target_window_handle = int(target_window_handle or 0)
        self.image_format = image_format.upper()
        self.jpeg_quality = max(20, min(95, int(jpeg_quality)))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def set_target_window_title(self, title: str) -> None:
        self.target_window_title = title.strip()
        self.target_window_handle = 0

    def capture(self) -> CaptureFrame:
        image_path = self._make_capture_path()

        if self.capture_mode == "region":
            self._capture_region(image_path, self.capture_region)
        elif self.target_window_handle or self.target_window_title:
            self._capture_window(image_path)
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

        if self.capture_mode == "region":
            safe_target = "region"
        elif self.target_window_title:
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

    def _capture_region(
        self,
        output_path: Path,
        region: tuple[int, int, int, int],
    ) -> None:
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
                self._save_image(image, output_path)
                return

        except Exception as exc:
            raise RuntimeError(f"Region capture failed: {exc}") from exc

    def _capture_window(self, output_path: Path) -> None:
        """
        Captures a window directly rather than reading its screen rectangle.

        This keeps working if the title changes and excludes windows covering
        the target. Some hardware-accelerated applications may still return a
        blank image because they do not support Windows direct-window capture.
        """
        try:
            import ctypes
            from PIL import ImageGrab

            handle = self.target_window_handle
            if not handle:
                handle = _find_window_handle(self.target_window_title)
                self.target_window_handle = handle

            user32 = ctypes.windll.user32
            if not user32.IsWindow(handle):
                raise RuntimeError(
                    "The selected window no longer exists; refresh and select it again."
                )
            if user32.IsIconic(handle):
                raise RuntimeError(
                    "The selected window is minimized and cannot be captured."
                )

            image = ImageGrab.grab(window=handle)
            if image.width <= 0 or image.height <= 0:
                raise RuntimeError("The selected window returned an empty image.")
            self._save_image(image, output_path)

        except Exception as exc:
            raise RuntimeError(f"Window capture failed: {exc}") from exc


def _find_window_handle(window_title: str) -> int:
    title_lower = window_title.casefold()
    matches = [
        handle
        for handle, title in list_windows()
        if title_lower in title.casefold()
    ]
    if not matches:
        raise RuntimeError(
            f"No window found with title containing: {window_title}"
        )
    return matches[0]


def list_windows() -> list[tuple[int, str]]:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    windows: set[tuple[int, str]] = set()

    enum_callback = ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        wintypes.HWND,
        wintypes.LPARAM,
    )

    @enum_callback
    def collect_window(handle, _extra) -> bool:
        if not user32.IsWindowVisible(handle):
            return True
        title_length = user32.GetWindowTextLengthW(handle)
        if title_length <= 0:
            return True
        title_buffer = ctypes.create_unicode_buffer(title_length + 1)
        user32.GetWindowTextW(handle, title_buffer, title_length + 1)
        title = title_buffer.value.strip()
        if title:
            windows.add((int(handle), title))
        return True

    if not user32.EnumWindows(collect_window, 0):
        raise RuntimeError("Windows could not enumerate open windows.")
    return sorted(windows, key=lambda item: item[1].casefold())


def list_window_titles() -> list[str]:
    return [title for _, title in list_windows()]


def main() -> None:
    print("Available windows:")
    for title in list_window_titles():
        print("-", title)

    service = CaptureService(output_dir=Path("logs/captures"))
    frame = service.capture()
    print(f"Captured: {frame.image_path.resolve()}")


if __name__ == "__main__":
    main()
