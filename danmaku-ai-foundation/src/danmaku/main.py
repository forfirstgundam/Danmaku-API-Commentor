from __future__ import annotations

import json
import random
import sys
import threading
import time
from datetime import datetime

from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import QApplication

from danmaku.api.llm_client import LLMClient
from danmaku.capture.capture_service import CaptureService
from danmaku.config import load_settings_from_env
from danmaku.models import AppSettings, CaptureFrame, CommentBatch
from danmaku.overlay.overlay_window import OverlayWindow
from danmaku.ui.settings_window import SettingsWindow


class AppSignals(QObject):
    frame_sampled = pyqtSignal(object)
    sample_error = pyqtSignal(str)
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

        # S: rewritten whole/canonical summary.
        self.previous_summary = ""

        # T history: recent current-situation snapshots only.
        self.situation_history: list[str] = []

        # Optional debug history of rewritten summaries.
        self.summary_versions: list[str] = []

        # Recent generated comments used to reduce repetition.
        self.recent_comment_history: list[str] = []

        # Reusable comments for API-failure fallback.
        self.fallback_comment_pool: list[str] = []

        # State for one streamed API response.
        self.stream_batch_started = False
        self.streamed_comments_current_batch: list[str] = []

        self.consecutive_api_failures = 0
        self.is_running = False
        self.is_busy = False
        self.is_sampling = False
        self.waiting_for_first_request = False
        self.frame_buffer: list[CaptureFrame] = []

        self.signals = AppSignals()
        self.signals.frame_sampled.connect(self._on_frame_sampled)
        self.signals.sample_error.connect(self._on_sample_error)
        self.signals.partial_comment_ready.connect(self._on_partial_comment_ready)
        self.signals.comments_ready.connect(self._on_comments_ready)
        self.signals.error.connect(self._on_error)

        self.capture_service = CaptureService(
            output_dir=self.settings.capture_output_dir,
            target_window_title=self.settings.target_window_title,
            image_format="JPEG",
            jpeg_quality=self.settings.sample_capture_jpeg_quality,
        )

        self.llm_client = self._build_llm_client()
        self.fallback_llm_client = self._build_fallback_llm_client()

        self.overlay = OverlayWindow(settings=self.settings)
        self.settings_window = SettingsWindow(settings=self.settings)

        self.settings_window.start_requested.connect(self.start)
        self.settings_window.stop_requested.connect(self.stop)

        self.capture_timer = QTimer()
        self.capture_timer.timeout.connect(self._trigger_capture_and_api)

        self.sample_timer = QTimer()
        self.sample_timer.timeout.connect(self._trigger_sample_capture)

    def show(self) -> None:
        self.settings_window.show()

    def start(self) -> None:
        self.settings_window.apply_to_settings()
        self.consecutive_api_failures = 0
        self.stream_batch_started = False
        self.streamed_comments_current_batch = []
        self.frame_buffer = []
        self.is_sampling = False
        # Single-frame mode preserves the old quick first request. Multi-frame
        # mode waits for the normal API interval so its first buffer can fill.
        self.waiting_for_first_request = not self.settings.use_multi_frame_context

        self._initialize_run_logging()

        self.capture_service = CaptureService(
            output_dir=self.settings.capture_output_dir,
            target_window_title=self.settings.target_window_title,
            image_format="JPEG",
            jpeg_quality=self.settings.sample_capture_jpeg_quality,
        )
        self.llm_client = self._build_llm_client()
        self.fallback_llm_client = self._build_fallback_llm_client()

        print("[app] starting")
        print(f"[app] dummy_api={self.settings.use_dummy_api}")
        print(f"[app] api_key_set={bool(self.settings.api_key)}")
        print(f"[app] api_provider={self.settings.api_provider}")
        print(f"[app] model={self.settings.model_name}")
        print(
            "[app] fallback_model="
            f"{self.settings.fallback_model_name or '(disabled)'}"
        )
        print(
            "[app] send_screenshot_to_api="
            f"{self.settings.send_screenshot_to_api}"
        )
        print(
            "[app] api_image_max_dimension="
            f"{self.settings.api_image_max_dimension}"
        )
        print(
            "[app] api_image_jpeg_quality="
            f"{self.settings.api_image_jpeg_quality}"
        )
        print(
            "[app] api_max_output_tokens="
            f"{self.settings.api_max_output_tokens}"
        )
        print(f"[app] use_streaming_api={self.settings.use_streaming_api}")
        print(
            "[app] multi_frame_context="
            f"{self.settings.use_multi_frame_context}, "
            f"sample_interval={self.settings.frame_sample_interval_seconds}s, "
            f"buffer_size={self.settings.frame_buffer_size}, "
            f"frames_per_request={self.settings.frames_per_request}"
        )
        print(
            "[app] history_image="
            f"{self.settings.history_image_max_dimension}px, "
            f"quality={self.settings.history_image_jpeg_quality}"
        )
        print(f"[app] save_api_images={self.settings.save_api_images}")
        print(
            "[app] capture_dir="
            f"{self.settings.capture_output_dir.resolve()}"
        )
        print(
            "[app] api_image_dir="
            f"{self.settings.api_image_output_dir.resolve()}"
        )
        print(
            "[app] comment_log_path="
            f"{self.settings.comment_log_path.resolve()}"
        )
        print(
            "[app] target_window="
            f"{self.settings.target_window_title or 'Full screen'}"
        )

        interval_ms = self.settings.capture_interval_seconds * 1000
        self.capture_timer.start(interval_ms)

        sample_interval_seconds = (
            self.settings.frame_sample_interval_seconds
            if self.settings.use_multi_frame_context
            else self.settings.capture_interval_seconds
        )
        self.sample_timer.setInterval(max(1, sample_interval_seconds) * 1000)

        self.overlay.show()
        self.is_running = True
        self.settings_window.set_running(True)

        # Minimize the settings window so it is less likely to be captured.
        self.settings_window.showMinimized()

        print(
            "[app] first frame sample scheduled after "
            f"{self.settings.first_capture_delay_ms} ms"
        )

        QTimer.singleShot(
            self.settings.first_capture_delay_ms,
            self._begin_sampling,
        )

    def stop(self) -> None:
        self.capture_timer.stop()
        self.sample_timer.stop()
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
            history_image_max_dimension=(
                self.settings.history_image_max_dimension
            ),
            history_image_jpeg_quality=(
                self.settings.history_image_jpeg_quality
            ),
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
            history_image_max_dimension=(
                self.settings.history_image_max_dimension
            ),
            history_image_jpeg_quality=(
                self.settings.history_image_jpeg_quality
            ),
            max_output_tokens=self.settings.api_max_output_tokens,
            save_api_images=self.settings.save_api_images,
            api_image_output_dir=self.settings.api_image_output_dir,
        )

    def _build_context_summary(self) -> str:
        """
        Build context sent to the LLM.

        Sends:
        - S(n-1): previous rewritten whole summary
        - recent T snapshots: current_situation from recent captures
        """

        parts: list[str] = []

        if self.previous_summary:
            parts.append(
                "Previous whole summary S(n-1):\n"
                f"{self.previous_summary}"
            )

        recent_situations = self.situation_history[-3:]

        if recent_situations:
            recent_text = "\n".join(
                f"T-{len(recent_situations) - index}: {situation}"
                for index, situation in enumerate(recent_situations)
            )

            parts.append(
                "Recent current-situation snapshots, oldest to newest:\n"
                f"{recent_text}"
            )

        return "\n\n".join(parts)

    def _trigger_sample_capture(self) -> None:
        if not self.is_running or self.is_sampling:
            return

        self.is_sampling = True

        thread = threading.Thread(
            target=self._sample_capture_worker,
            daemon=True,
        )
        thread.start()

    def _begin_sampling(self) -> None:
        if not self.is_running:
            return

        self._trigger_sample_capture()
        self.sample_timer.start()

    def _sample_capture_worker(self) -> None:
        try:
            capture_started = time.perf_counter()
            frame = self.capture_service.capture()
            capture_duration_sec = round(
                time.perf_counter() - capture_started,
                3,
            )
            self.signals.frame_sampled.emit(
                {
                    "frame": frame,
                    "capture_duration_sec": capture_duration_sec,
                }
            )
        except Exception as exc:
            self.signals.sample_error.emit(str(exc))

    def _on_frame_sampled(self, payload: object) -> None:
        self.is_sampling = False
        data = payload if isinstance(payload, dict) else {}
        frame = data.get("frame")

        if not isinstance(frame, CaptureFrame):
            print("[capture] invalid sampled frame")
            return

        self.frame_buffer.append(frame)
        buffer_size = max(1, self.settings.frame_buffer_size)
        self.frame_buffer = self.frame_buffer[-buffer_size:]

        image_size_kb = round(frame.image_path.stat().st_size / 1024, 1)
        print(
            "[capture] sampled "
            f"{frame.image_path} ({image_size_kb} KB) "
            f"in {data.get('capture_duration_sec')}s; "
            f"buffer={len(self.frame_buffer)}/{buffer_size}"
        )

        if self.waiting_for_first_request and self.is_running:
            self.waiting_for_first_request = False
            QTimer.singleShot(0, self._trigger_capture_and_api)

    def _on_sample_error(self, message: str) -> None:
        self.is_sampling = False
        print(f"[capture] sample failed: {message}")

    def _select_frames_for_request(self) -> list[CaptureFrame]:
        if not self.frame_buffer:
            return []

        if not self.settings.use_multi_frame_context:
            return [self.frame_buffer[-1]]

        requested_count = max(1, self.settings.frames_per_request)
        frame_count = min(requested_count, len(self.frame_buffer))

        if frame_count == len(self.frame_buffer):
            return list(self.frame_buffer)

        if frame_count == 1:
            return [self.frame_buffer[-1]]

        last_index = len(self.frame_buffer) - 1
        indices = [
            round(step * last_index / (frame_count - 1))
            for step in range(frame_count)
        ]
        return [self.frame_buffer[index] for index in indices]

    def _trigger_capture_and_api(self) -> None:
        if not self.is_running or self.is_busy:
            return

        selected_frames = self._select_frames_for_request()
        if not selected_frames:
            print("[api] skipped request: no sampled frames available")
            self._trigger_sample_capture()
            return

        self.is_busy = True
        self.stream_batch_started = False
        self.streamed_comments_current_batch = []

        frame_span_sec = round(
            selected_frames[-1].timestamp - selected_frames[0].timestamp,
            3,
        )
        print(
            "[context] selected frames: "
            f"count={len(selected_frames)}, span={frame_span_sec}s, "
            f"ages={[round(selected_frames[-1].timestamp - item.timestamp, 2) for item in selected_frames]}"
        )

        thread = threading.Thread(
            target=self._capture_and_generate_worker,
            args=(selected_frames,),
            daemon=True,
        )
        thread.start()

    def _capture_and_generate_worker(
        self,
        frames: list[CaptureFrame],
    ) -> None:
        try:
            worker_started = time.perf_counter()
            frame = frames[-1]
            context_frames = frames[:-1]

            latest_frame_age_at_request_sec = round(
                max(0.0, time.time() - frame.timestamp),
                3,
            )
            api_started = time.perf_counter()
            context_for_api = self._build_context_summary()

            summary_before_request = self.previous_summary
            situations_before_request = self.situation_history[-3:]
            recent_comments_for_api = self.recent_comment_history[-12:]

            first_partial_comment_at: float | None = None
            streamed_comment_count = 0

            print(
                "[context] "
                f"summary_chars={len(self.previous_summary)}, "
                f"situation_count={len(self.situation_history)}, "
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
                context_frames=context_frames,
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
                    context_frames=context_frames,
                    use_streaming=False,
                    on_comment=None,
                )

                retry_duration_sec = round(
                    time.perf_counter() - retry_started,
                    3,
                )
                retry_used = True

                if retry_batch.is_error:
                    retry_error_message = retry_batch.error_message
                    print(
                        "[api] primary retry failed: "
                        f"{retry_error_message}"
                    )
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
                    f"{self.settings.fallback_model_name}: "
                    f"{batch.error_message}"
                )

                fallback_batch = self.fallback_llm_client.generate_comments(
                    frame=frame,
                    previous_summary=context_for_api,
                    previous_comments=recent_comments_for_api,
                    context_frames=context_frames,
                    use_streaming=False,
                    on_comment=None,
                )

                fallback_duration_sec = round(
                    time.perf_counter() - fallback_started,
                    3,
                )
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
                "capture_duration_sec": None,
                "comment_after_capture_sec": round(
                    time.time() - frame.timestamp,
                    3,
                ),
                "api_duration_sec": round(
                    api_finished - api_started,
                    3,
                ),
                "total_worker_duration_sec": round(
                    api_finished - worker_started,
                    3,
                ),
                "first_streamed_comment_sec": (
                    round(first_partial_comment_at - api_started, 3)
                    if first_partial_comment_at is not None
                    else None
                ),
                "streamed_comment_count": streamed_comment_count,
                "frames_sent_count": len(frames),
                "frames_sent_span_sec": round(
                    frame.timestamp - frames[0].timestamp,
                    3,
                ),
                "latest_frame_age_at_request_sec": round(
                    latest_frame_age_at_request_sec,
                    3,
                ),
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
                f"frames={metrics['frames_sent_count']}, "
                f"frame_span={metrics['frames_sent_span_sec']}s, "
                f"after_capture_to_comments="
                f"{metrics['comment_after_capture_sec']}s, "
                f"first_streamed_comment="
                f"{metrics['first_streamed_comment_sec']}s, "
                f"fallback_used={metrics['fallback_used']}, "
                f"total={metrics['total_worker_duration_sec']}s"
            )

            self.signals.comments_ready.emit(
                {
                    "frame": frame,
                    "frames_sent": frames,
                    "batch": batch,
                    "metrics": metrics,
                    "context_sent": context_for_api,
                    "summary_before_request": summary_before_request,
                    "situations_before_request": situations_before_request,
                    "recent_comments_sent": recent_comments_for_api,
                    "streamed_comment_count": streamed_comment_count,
                }
            )

        except Exception as exc:
            self.signals.error.emit(str(exc))

    def _on_comments_ready(self, payload: object) -> None:
        self.is_busy = False

        data = payload if isinstance(payload, dict) else {}

        frame = data.get("frame")
        frames_sent = data.get("frames_sent", [])
        batch = data.get("batch")
        metrics = data.get("metrics", {})
        context_sent = data.get("context_sent", "")
        summary_before_request = data.get("summary_before_request", "")
        situations_before_request = data.get(
            "situations_before_request",
            [],
        )
        recent_comments_sent = data.get("recent_comments_sent", [])
        streamed_comment_count = data.get("streamed_comment_count", 0)

        if not isinstance(frame, CaptureFrame):
            print("[app] invalid frame payload")
            self._reset_stream_batch_state()
            return

        if not isinstance(batch, CommentBatch):
            print("[app] invalid comment batch payload")
            self._reset_stream_batch_state()
            return

        if batch.is_error:
            self._handle_api_failure(
                batch.error_message,
                streamed_comment_count,
            )

            self._reset_stream_batch_state()

            if self.settings.save_comments:
                self._save_comment_batch(
                    frame=frame,
                    batch=batch,
                    metrics=metrics,
                    context_sent=context_sent,
                    summary_before_request=summary_before_request,
                    situations_before_request=(
                        situations_before_request
                        if isinstance(situations_before_request, list)
                        else []
                    ),
                    recent_comments_sent=(
                        recent_comments_sent
                        if isinstance(recent_comments_sent, list)
                        else []
                    ),
                    frames_sent=(
                        frames_sent if isinstance(frames_sent, list) else []
                    ),
                )

            return

        if self.consecutive_api_failures:
            print(
                "[api] recovered after "
                f"{self.consecutive_api_failures} consecutive failure(s)"
            )

        self.consecutive_api_failures = 0

        clean_summary = batch.summary.strip()
        scene_changed = clean_summary.startswith("[SCENE_CHANGE]")

        if scene_changed:
            clean_summary = clean_summary.removeprefix(
                "[SCENE_CHANGE]"
            ).strip()

            # Old per-frame snapshots should not leak into the new scene.
            self.situation_history.clear()
            self.previous_summary = ""

            print(
                "[context] scene change detected: "
                "reset previous summary and situation history"
            )

        if clean_summary:
            # S(n): rewritten whole summary replaces S(n-1).
            self.previous_summary = clean_summary

            # Keep old S versions only for debugging and logging.
            self.summary_versions.append(clean_summary)
            self.summary_versions = self.summary_versions[-8:]

        if batch.current_situation:
            clean_situation = batch.current_situation.strip()

            if clean_situation:
                # T(n): append the current factual snapshot.
                self.situation_history.append(clean_situation)
                self.situation_history = self.situation_history[-4:]

        print(
            "[context] updated: "
            f"summary_chars={len(self.previous_summary)}, "
            f"situation_count={len(self.situation_history)}"
        )

        if self.stream_batch_started:
            self.overlay.finish_streamed_batch(
                batch,
                self.streamed_comments_current_batch,
            )
        else:
            # Non-streaming response, retry, fallback model, or a response
            # where no complete comment was parsed before completion.
            self.overlay.add_comment_batch(batch)

        self._reset_stream_batch_state()
        self._remember_generated_comments(batch)

        if self.settings.save_comments:
            self._save_comment_batch(
                frame=frame,
                batch=batch,
                metrics=metrics,
                context_sent=context_sent,
                summary_before_request=summary_before_request,
                situations_before_request=(
                    situations_before_request
                    if isinstance(situations_before_request, list)
                    else []
                ),
                recent_comments_sent=(
                    recent_comments_sent
                    if isinstance(recent_comments_sent, list)
                    else []
                ),
                frames_sent=(
                    frames_sent if isinstance(frames_sent, list) else []
                ),
            )

    def _reset_stream_batch_state(self) -> None:
        self.stream_batch_started = False
        self.streamed_comments_current_batch = []

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
        streamed_comment_count: int,
    ) -> None:
        self.consecutive_api_failures += 1

        print(
            "[api] no overlay update: "
            f"{error_message} "
            f"(consecutive_failures={self.consecutive_api_failures})"
        )

        if (
            self.consecutive_api_failures < 3
            and streamed_comment_count == 0
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

        print(
            "[fallback] showing "
            f"{len(fallback_comments)} cached comment(s)"
        )

        # A fallback result is also a new batch, so remove stale pending
        # comments from the previous successful response.
        self.overlay.begin_streamed_batch()

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

    def _save_comment_batch(
        self,
        frame: CaptureFrame,
        batch: CommentBatch,
        metrics: dict | None = None,
        context_sent: str = "",
        summary_before_request: str = "",
        situations_before_request: list[str] | None = None,
        recent_comments_sent: list[str] | None = None,
        frames_sent: list[CaptureFrame] | None = None,
    ) -> None:
        self.settings.comment_log_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        log_path = self.settings.comment_log_path

        record = {
            "logged_at": datetime.now().isoformat(timespec="seconds"),
            "capture_timestamp": frame.timestamp,
            "image_path": str(frame.image_path),
            "ocr_text": frame.ocr_text,
            "comments": batch.comments,
            "long_comments": batch.long_comments,

            # Context sent before this response.
            "summary_before_request": summary_before_request,
            "situations_before_request": situations_before_request or [],
            "recent_comments_sent": recent_comments_sent or [],
            "context_sent": context_sent,
            "frames_sent": [
                {
                    "image_path": str(item.image_path),
                    "timestamp": item.timestamp,
                    "age_from_latest_sec": round(
                        max(0.0, frame.timestamp - item.timestamp),
                        3,
                    ),
                    "is_latest": item.image_path == frame.image_path,
                }
                for item in (frames_sent or [frame])
                if isinstance(item, CaptureFrame)
            ],

            # Response.
            "summary": batch.summary,
            "current_situation": batch.current_situation,
            "is_error": batch.is_error,
            "error_message": batch.error_message,
            "scene_change_detected": batch.summary.strip().startswith(
                "[SCENE_CHANGE]"
            ),

            # State after handling the response.
            "consecutive_api_failures": self.consecutive_api_failures,
            "summary_versions": self.summary_versions[-3:],
            "situation_history": self.situation_history[-3:],
            "recent_comment_history": self.recent_comment_history[-12:],

            # Configuration and timings.
            "used_dummy_api": self.settings.use_dummy_api,
            "api_provider": self.settings.api_provider,
            "model": self.settings.model_name,
            "fallback_model": self.settings.fallback_model_name,
            "send_screenshot_to_api": (
                self.settings.send_screenshot_to_api
            ),
            "api_image_max_dimension": (
                self.settings.api_image_max_dimension
            ),
            "api_image_jpeg_quality": (
                self.settings.api_image_jpeg_quality
            ),
            "use_multi_frame_context": (
                self.settings.use_multi_frame_context
            ),
            "frame_sample_interval_seconds": (
                self.settings.frame_sample_interval_seconds
            ),
            "frame_buffer_size": self.settings.frame_buffer_size,
            "frames_per_request": self.settings.frames_per_request,
            "history_image_max_dimension": (
                self.settings.history_image_max_dimension
            ),
            "history_image_jpeg_quality": (
                self.settings.history_image_jpeg_quality
            ),
            "api_max_output_tokens": self.settings.api_max_output_tokens,
            "use_streaming_api": self.settings.use_streaming_api,
            "save_api_images": self.settings.save_api_images,
            "timing": metrics or {},
        }

        with log_path.open("a", encoding="utf-8") as file:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

        print(f"[comments] saved {log_path}")

    def _on_partial_comment_ready(self, payload: object) -> None:
        data = payload if isinstance(payload, dict) else {}
        text = data.get("text", "")
        elapsed_sec = data.get("elapsed_sec")

        if not isinstance(text, str):
            return

        clean = text.strip()

        if not clean:
            return

        is_first_comment = not self.stream_batch_started

        if is_first_comment:
            self.overlay.begin_streamed_batch()
            self.stream_batch_started = True

        self.streamed_comments_current_batch.append(clean)

        print(f"[stream] comment received at {elapsed_sec}s: {clean}")

        # Preserve main's behavior: the first comment in a new batch may
        # appear immediately, while later comments remain paced by the
        # normal overlay timer.
        try:
            self.overlay.add_streamed_comment(
                clean,
                spawn_immediately=is_first_comment,
            )
        except TypeError:
            # Compatibility with an OverlayWindow version whose method only
            # accepts the text argument.
            self.overlay.add_streamed_comment(clean)

    def _initialize_run_logging(self) -> None:
        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")

        self.settings.run_log_dir = self.settings.log_root_dir / run_id
        self.settings.capture_output_dir = (
            self.settings.run_log_dir / "captures"
        )
        self.settings.api_image_output_dir = (
            self.settings.run_log_dir / "api_images"
        )
        self.settings.comment_log_path = (
            self.settings.run_log_dir / "comments.jsonl"
        )

        self.settings.capture_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if self.settings.save_api_images:
            self.settings.api_image_output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

        print(f"[log] run_id={run_id}")
        print(f"[log] run_dir={self.settings.run_log_dir.resolve()}")
        print(
            "[log] capture_dir="
            f"{self.settings.capture_output_dir.resolve()}"
        )
        print(
            "[log] api_image_dir="
            f"{self.settings.api_image_output_dir.resolve()}"
        )
        print(
            "[log] comments="
            f"{self.settings.comment_log_path.resolve()}"
        )

    def _on_error(self, message: str) -> None:
        self.is_busy = False
        self._reset_stream_batch_state()
        print(f"[app] error: {message}")


def main() -> None:
    app = QApplication(sys.argv)
    settings = load_settings_from_env()
    danmaku_app = DanmakuApp(settings=settings)
    danmaku_app.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
