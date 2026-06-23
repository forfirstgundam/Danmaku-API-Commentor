from __future__ import annotations

import json
import random
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import QApplication

from danmaku.api.llm_client import LLMClient
from danmaku.capture.capture_service import CaptureService
from danmaku.config import load_settings_from_env
from danmaku.models import AppSettings, CaptureFrame, CommentBatch
from danmaku.overlay.overlay_window import OverlayWindow
from danmaku.ui.settings_window import SettingsWindow


class AppSignals(QObject):
    partial_comment_ready = pyqtSignal(object)
    comments_ready = pyqtSignal(object)
    error = pyqtSignal(str)


class DanmakuApp:
    """
    Application coordinator.

    This class connects:
    - UI
    - capture
    - API
    - overlay
    - testing logs
    """

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.previous_summary = ""
        self.summary_history: list[str] = []
        self.recent_comment_history: list[str] = []
        self.fallback_comment_pool: list[str] = []
        self.consecutive_api_failures = 0
        self.is_running = False
        self.is_busy = False

        self.signals = AppSignals()
        self.signals.partial_comment_ready.connect(self._on_partial_comment_ready)
        self.signals.comments_ready.connect(self._on_comments_ready)
        self.signals.error.connect(self._on_error)

        self.capture_service = CaptureService(
            output_dir=self.settings.capture_output_dir,
            target_window_title=self.settings.target_window_title,
        )

        self.llm_client = self._build_llm_client()
        self.fallback_llm_client = self._build_fallback_llm_client()

        self.overlay = OverlayWindow(settings=self.settings)
        self.settings_window = SettingsWindow(settings=self.settings)

        self.settings_window.start_requested.connect(self.start)
        self.settings_window.stop_requested.connect(self.stop)

        self.capture_timer = QTimer()
        self.capture_timer.timeout.connect(self._trigger_capture_and_api)

    def show(self) -> None:
        self.settings_window.show()

    def start(self) -> None:
        self.settings_window.apply_to_settings()
        self.consecutive_api_failures = 0
        self._initialize_run_logging()
        self.capture_service = CaptureService(
            output_dir=self.settings.capture_output_dir,
            target_window_title=self.settings.target_window_title,
        )
        self.llm_client = self._build_llm_client()
        self.fallback_llm_client = self._build_fallback_llm_client()

        print("[app] starting")
        print(f"[app] dummy_api={self.settings.use_dummy_api}")
        print(f"[app] api_key_set={bool(self.settings.api_key)}")
        print(f"[app] api_provider={self.settings.api_provider}")
        print(f"[app] model={self.settings.model_name}")
        print(f"[app] fallback_model={self.settings.fallback_model_name or '(disabled)'}")
        print(f"[app] send_screenshot_to_api={self.settings.send_screenshot_to_api}")
        print(f"[app] api_image_max_dimension={self.settings.api_image_max_dimension}")
        print(f"[app] api_image_jpeg_quality={self.settings.api_image_jpeg_quality}")
        print(f"[app] api_max_output_tokens={self.settings.api_max_output_tokens}")
        print(f"[app] use_streaming_api={self.settings.use_streaming_api}")
        print(f"[app] save_api_images={self.settings.save_api_images}")
        print(f"[app] capture_dir={self.settings.capture_output_dir.resolve()}")
        print(f"[app] api_image_dir={self.settings.api_image_output_dir.resolve()}")
        print(f"[app] comment_log_path={self.settings.comment_log_path.resolve()}")
        print(f"[app] target_window={self.settings.target_window_title or 'Full screen'}")

        interval_ms = self.settings.capture_interval_seconds * 1000
        self.capture_timer.start(interval_ms)

        self.overlay.show()
        self.is_running = True
        self.settings_window.set_running(True)

        # Minimize the settings window so it is less likely to be captured.
        self.settings_window.showMinimized()

        print(
            "[app] first capture scheduled after "
            f"{self.settings.first_capture_delay_ms} ms"
        )

        QTimer.singleShot(
            self.settings.first_capture_delay_ms,
            self._trigger_capture_and_api,
        )

    def stop(self) -> None:
        self.capture_timer.stop()
        self.overlay.hide()
        self.is_running = False
        self.settings_window.set_running(False)
        print("[app] stopped")

    def _build_llm_client(self) -> LLMClient:
        return LLMClient(
            api_key=self.settings.api_key,
            api_provider=self.settings.api_provider,
            model_name=self.settings.model_name,
            use_dummy_api=self.settings.use_dummy_api,
            send_screenshot=self.settings.send_screenshot_to_api,
            image_max_dimension=self.settings.api_image_max_dimension,
            image_jpeg_quality=self.settings.api_image_jpeg_quality,
            max_output_tokens=self.settings.api_max_output_tokens,
            save_api_images=self.settings.save_api_images,
            api_image_output_dir=self.settings.api_image_output_dir,
        )

    def _build_fallback_llm_client(self) -> LLMClient | None:
        fallback_model_name = self.settings.fallback_model_name.strip()

        if not fallback_model_name:
            return None

        if fallback_model_name == self.settings.model_name:
            return None

        return LLMClient(
            api_key=self.settings.api_key,
            api_provider=self.settings.api_provider,
            model_name=fallback_model_name,
            use_dummy_api=self.settings.use_dummy_api,
            send_screenshot=self.settings.send_screenshot_to_api,
            image_max_dimension=self.settings.api_image_max_dimension,
            image_jpeg_quality=self.settings.api_image_jpeg_quality,
            max_output_tokens=self.settings.api_max_output_tokens,
            save_api_images=self.settings.save_api_images,
            api_image_output_dir=self.settings.api_image_output_dir,
        )

    def _build_context_summary(self) -> str:
        """
        Build context sent to the LLM.

        Includes:
        - rolling summary of the overall situation
        - last 4 scene summaries
        """

        parts: list[str] = []

        if self.previous_summary:
            parts.append(
                "Overall context so far:\n"
                f"{self.previous_summary}"
            )

        recent = self.summary_history[-2:]

        if recent:
            recent_text = "\n".join(
                f"{index + 1}. {summary}"
                for index, summary in enumerate(recent)
            )

            parts.append(
                "Recent scene history, oldest to newest:\n"
                f"{recent_text}"
            )

        return "\n\n".join(parts)

    def _build_rolling_summary(self) -> str:
        """
        Build a bounded rolling summary from recent scene summaries.

        This is not perfect semantic compression, but it keeps enough continuity
        without sending unlimited history.
        """

        recent = self.summary_history[-4:]

        if not recent:
            return ""

        joined = " ".join(recent)

        max_chars = 600

        if len(joined) <= max_chars:
            return joined

        return joined[-max_chars:]

    def _trigger_capture_and_api(self) -> None:
        if not self.is_running or self.is_busy:
            return

        self.is_busy = True
        thread = threading.Thread(
            target=self._capture_and_generate_worker, daemon=True)
        thread.start()

    def _capture_and_generate_worker(self) -> None:
        try:
            worker_started = time.perf_counter()

            capture_started = time.perf_counter()
            frame = self.capture_service.capture()
            capture_finished = time.perf_counter()

            image_size_kb = round(frame.image_path.stat().st_size / 1024, 1)
            print(f"[capture] saved {frame.image_path} ({image_size_kb} KB)")

            api_started = time.perf_counter()
            context_for_api = self._build_context_summary()
            recent_comments_for_api = self.recent_comment_history[-12:]
            first_partial_comment_at: float | None = None
            streamed_comment_count = 0

            print(
                "[context] "
                f"rolling_summary_chars={len(self.previous_summary)}, "
                f"history_count={len(self.summary_history)}, "
                f"recent_comment_count={len(recent_comments_for_api)}, "
                f"context_sent_chars={len(context_for_api)}"
            )

            def on_streamed_comment(comment: str) -> None:
                nonlocal first_partial_comment_at, streamed_comment_count

                now = time.perf_counter()
                if first_partial_comment_at is None:
                    first_partial_comment_at = now

                streamed_comment_count += 1
                self.signals.partial_comment_ready.emit(
                    {
                        "text": comment,
                        "elapsed_sec": round(now - api_started, 3),
                    }
                )

            batch = self.llm_client.generate_comments(
                frame=frame,
                previous_summary=context_for_api,
                previous_comments=recent_comments_for_api,
                use_streaming=self.settings.use_streaming_api,
                on_comment=on_streamed_comment,
            )

            retry_used = False
            retry_error_message = ""
            retry_duration_sec = None

            if batch.is_error and streamed_comment_count == 0:
                retry_started = time.perf_counter()
                print(
                    "[api] primary failed before streaming comments; "
                    f"retrying {self.settings.model_name} without streaming: "
                    f"{batch.error_message}"
                )

                retry_batch = self.llm_client.generate_comments(
                    frame=frame,
                    previous_summary=context_for_api,
                    previous_comments=recent_comments_for_api,
                    use_streaming=False,
                    on_comment=None,
                )

                retry_duration_sec = round(
                    time.perf_counter() - retry_started, 3)
                retry_used = True

                if retry_batch.is_error:
                    retry_error_message = retry_batch.error_message
                    print(f"[api] primary retry failed: {retry_error_message}")
                else:
                    print(
                        "[api] primary retry succeeded: "
                        f"{self.settings.model_name} "
                        f"in {retry_duration_sec}s"
                    )
                    batch = retry_batch

            fallback_used = False
            fallback_error_message = ""
            fallback_duration_sec = None

            if (
                batch.is_error
                and streamed_comment_count == 0
                and self.fallback_llm_client is not None
            ):
                fallback_started = time.perf_counter()
                print(
                    "[api] primary failed; trying fallback model "
                    f"{self.settings.fallback_model_name}: {batch.error_message}"
                )

                fallback_batch = self.fallback_llm_client.generate_comments(
                    frame=frame,
                    previous_summary=context_for_api,
                    previous_comments=recent_comments_for_api,
                    use_streaming=False,
                    on_comment=None,
                )

                fallback_duration_sec = round(
                    time.perf_counter() - fallback_started, 3)
                fallback_used = not fallback_batch.is_error

                if fallback_batch.is_error:
                    fallback_error_message = fallback_batch.error_message
                    print(
                        "[api] fallback failed: "
                        f"{fallback_error_message}"
                    )
                else:
                    print(
                        "[api] fallback succeeded: "
                        f"{self.settings.fallback_model_name} "
                        f"in {fallback_duration_sec}s"
                    )
                    batch = fallback_batch

            api_finished = time.perf_counter()

            metrics = {
                "capture_duration_sec": round(capture_finished - capture_started, 3),
                "comment_after_capture_sec": round(api_finished - capture_finished, 3),
                "api_duration_sec": round(api_finished - api_started, 3),
                "total_worker_duration_sec": round(api_finished - worker_started, 3),
                "first_streamed_comment_sec": (
                    round(first_partial_comment_at - api_started, 3)
                    if first_partial_comment_at is not None
                    else None
                ),
                "streamed_comment_count": streamed_comment_count,
                "retry_used": retry_used,
                "retry_duration_sec": retry_duration_sec,
                "retry_error_message": retry_error_message,
                "fallback_used": fallback_used,
                "fallback_model": (
                    self.settings.fallback_model_name
                    if self.fallback_llm_client is not None
                    else ""
                ),
                "fallback_duration_sec": fallback_duration_sec,
                "fallback_error_message": fallback_error_message,
            }

            print(
                "[timing] "
                f"capture={metrics['capture_duration_sec']}s, "
                f"after_capture_to_comments={metrics['comment_after_capture_sec']}s, "
                f"first_streamed_comment={metrics['first_streamed_comment_sec']}s, "
                f"fallback_used={metrics['fallback_used']}, "
                f"total={metrics['total_worker_duration_sec']}s"
            )

            self.signals.comments_ready.emit(
                {
                    "frame": frame,
                    "batch": batch,
                    "metrics": metrics,
                    "context_sent": context_for_api,
                    "recent_comments_sent": recent_comments_for_api,
                    "comments_displayed_during_stream": streamed_comment_count,
                }
            )

        except Exception as exc:
            self.signals.error.emit(str(exc))

    def _on_comments_ready(self, payload: object) -> None:
        self.is_busy = False

        data = payload if isinstance(payload, dict) else {}
        frame = data.get("frame")
        batch = data.get("batch")
        metrics = data.get("metrics", {})
        context_sent = data.get("context_sent", "")
        recent_comments_sent = data.get("recent_comments_sent", [])
        comments_displayed_during_stream = data.get(
            "comments_displayed_during_stream", 0)

        if not isinstance(frame, CaptureFrame):
            print("[app] invalid frame payload")
            return
        
        if not isinstance(batch, CommentBatch):
            print("[app] invalid comment batch payload")
            return

        if batch.is_error:
            self._handle_api_failure(
                batch.error_message,
                comments_displayed_during_stream,
            )

            if self.settings.save_comments:
                self._save_comment_batch(
                    frame,
                    batch,
                    metrics,
                    context_sent,
                    recent_comments_sent if isinstance(recent_comments_sent, list) else [],
                )
            return

        if self.consecutive_api_failures:
            print(
                "[api] recovered after "
                f"{self.consecutive_api_failures} consecutive failure(s)"
            )
        self.consecutive_api_failures = 0

        if batch.summary:
            clean_summary = batch.summary.strip()
            scene_changed = clean_summary.startswith("[SCENE_CHANGE]")

            if scene_changed:
                clean_summary = clean_summary.removeprefix(
                    "[SCENE_CHANGE]").strip()
                self.summary_history.clear()
                self.previous_summary = ""
                print("[context] scene change detected: reset summary history")

            self.summary_history.append(clean_summary)
            self.summary_history = self.summary_history[-4:]

            self.previous_summary = self._build_rolling_summary()

            print(
                "[context] updated: "
                f"history_count={len(self.summary_history)}, "
                f"rolling_summary_chars={len(self.previous_summary)}"
            )

        if comments_displayed_during_stream:
            print(
                "[overlay] final batch received after streaming: "
                f"already_displayed={comments_displayed_during_stream}"
            )
        else:
            self.overlay.add_comment_batch(batch)

        self._remember_generated_comments(batch)

        if self.settings.save_comments:
            self._save_comment_batch(
                frame,
                batch,
                metrics,
                context_sent,
                recent_comments_sent if isinstance(recent_comments_sent, list) else [],
            )

    def _remember_generated_comments(self, batch: CommentBatch) -> None:
        new_comments = [
            comment.strip()
            for comment in [*batch.comments, *batch.long_comments]
            if comment.strip()
        ]

        if not new_comments:
            return

        self.recent_comment_history.extend(new_comments)
        self.recent_comment_history = self.recent_comment_history[-24:]
        self._remember_fallback_candidates(batch.comments)

    def _remember_fallback_candidates(self, comments: list[str]) -> None:
        for comment in comments:
            clean = comment.strip()

            if not self._is_reusable_fallback_comment(clean):
                continue

            if clean in self.fallback_comment_pool:
                self.fallback_comment_pool.remove(clean)

            self.fallback_comment_pool.append(clean)

        self.fallback_comment_pool = self.fallback_comment_pool[-30:]

    @staticmethod
    def _is_reusable_fallback_comment(comment: str) -> bool:
        if not 2 <= len(comment) <= 18:
            return False

        if any(character.isdigit() for character in comment):
            return False

        if any("a" <= character.lower() <= "z" for character in comment):
            return False

        if comment.count(" ") >= 3:
            return False

        specific_terms = [
            "유튜브",
            "영상",
            "방송",
            "스트리머",
            "롤",
            "리그",
            "야구",
            "메이플",
            "스타듀",
            "슬더스",
            "고독",
            "미식",
            "메뉴",
            "보스",
            "카드",
            "대사",
        ]

        return not any(term in comment for term in specific_terms)

    def _handle_api_failure(
        self,
        error_message: str,
        comments_displayed_during_stream: int,
    ) -> None:
        self.consecutive_api_failures += 1

        print(
            "[api] no overlay update: "
            f"{error_message} "
            f"(consecutive_failures={self.consecutive_api_failures})"
        )

        if (
            self.consecutive_api_failures < 3
            and comments_displayed_during_stream == 0
        ):
            self._show_fallback_comments()

        if self.consecutive_api_failures >= 3:
            print("[api] stopping after 3 consecutive failures")
            self.stop()
            self._bring_settings_window_to_front()

    def _show_fallback_comments(self) -> None:
        fallback_comments = self._select_fallback_comments()

        if not fallback_comments:
            print("[fallback] no reusable comments available")
            return

        print(f"[fallback] showing {len(fallback_comments)} cached comment(s)")

        for comment in fallback_comments:
            self.overlay.add_streamed_comment(comment)

    def _select_fallback_comments(self) -> list[str]:
        default_comments = [
            "ㅋㅋㅋ",
            "오",
            "뭐야ㅋㅋ",
            "이건 좀 웃기네",
            "아니ㅋㅋ",
        ]

        candidates = self.fallback_comment_pool or default_comments
        count = min(3, len(candidates))

        return random.sample(candidates, count)

    def _bring_settings_window_to_front(self) -> None:
        self.settings_window.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self.settings_window.showNormal()
        self.settings_window.raise_()
        self.settings_window.activateWindow()
        QTimer.singleShot(1500, self._release_settings_window_topmost)

    def _release_settings_window_topmost(self) -> None:
        self.settings_window.setWindowFlag(Qt.WindowStaysOnTopHint, False)
        self.settings_window.showNormal()

    def _save_comment_batch(self,
                            frame: CaptureFrame,
                            batch: CommentBatch,
                            metrics: dict | None = None,
                            context_sent: str = "",
                            recent_comments_sent: list[str] | None = None,
                            ) -> None:
        self.settings.comment_log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path = self.settings.comment_log_path   

        record = {
            "logged_at": datetime.now().isoformat(timespec="seconds"),
            "capture_timestamp": frame.timestamp,
            "image_path": str(frame.image_path),
            "ocr_text": frame.ocr_text,
            "comments": batch.comments,
            "long_comments": batch.long_comments,
            "summary": batch.summary,
            "is_error": batch.is_error,
            "error_message": batch.error_message,
            "consecutive_api_failures": self.consecutive_api_failures,
            "summary_history": self.summary_history[-2:],
            "scene_change_detected": batch.summary.strip().startswith(
                "[SCENE_CHANGE]"),
            "recent_comments_sent": recent_comments_sent or [],
            "recent_comment_history": self.recent_comment_history[-12:],
            "context_sent": context_sent,
            "used_dummy_api": self.settings.use_dummy_api,
            "api_provider": self.settings.api_provider,
            "model": self.settings.model_name,
            "fallback_model": self.settings.fallback_model_name,
            "send_screenshot_to_api": self.settings.send_screenshot_to_api,
            "api_image_max_dimension": self.settings.api_image_max_dimension,
            "api_image_jpeg_quality": self.settings.api_image_jpeg_quality,
            "api_max_output_tokens": self.settings.api_max_output_tokens,
            "use_streaming_api": self.settings.use_streaming_api,
            "save_api_images": self.settings.save_api_images,
            "timing": metrics or {},
        }

        with log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"[comments] saved {log_path}")

    def _on_partial_comment_ready(self, payload: object) -> None:
        data = payload if isinstance(payload, dict) else {}
        text = data.get("text", "")
        elapsed_sec = data.get("elapsed_sec")

        if not isinstance(text, str) or not text.strip():
            return

        print(f"[stream] comment at {elapsed_sec}s: {text}")
        self.overlay.add_streamed_comment(text)

    def _initialize_run_logging(self) -> None:
        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")

        self.settings.run_log_dir = self.settings.log_root_dir / run_id
        self.settings.capture_output_dir = self.settings.run_log_dir / "captures"
        self.settings.api_image_output_dir = self.settings.run_log_dir / "api_images"
        self.settings.comment_log_path = self.settings.run_log_dir / "comments.jsonl"

        self.settings.capture_output_dir.mkdir(parents=True, exist_ok=True)
        if self.settings.save_api_images:
            self.settings.api_image_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[log] run_id={run_id}")
        print(f"[log] run_dir={self.settings.run_log_dir.resolve()}")
        print(
            f"[log] capture_dir={self.settings.capture_output_dir.resolve()}")
        print(
            f"[log] api_image_dir={self.settings.api_image_output_dir.resolve()}")
        print(f"[log] comments={self.settings.comment_log_path.resolve()}")

    def _on_error(self, message: str) -> None:
        self.is_busy = False
        print(f"[app] error: {message}")


def main() -> None:
    load_dotenv()
    
    app = QApplication(sys.argv)
    settings = load_settings_from_env()
    danmaku_app = DanmakuApp(settings=settings)
    danmaku_app.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
