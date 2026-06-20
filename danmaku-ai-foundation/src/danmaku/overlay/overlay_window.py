from __future__ import annotations

import random
import sys
from collections import deque
from dataclasses import dataclass
from typing import Deque

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QFontMetrics, QPainter
from PyQt5.QtWidgets import QApplication, QWidget

from danmaku.models import AppSettings, CommentBatch


@dataclass
class MovingComment:
    text: str
    x: float
    y: float
    speed: float
    width: int
    lane_index: int


class OverlayWindow(QWidget):
    """
    Lane-based optimized danmaku overlay.

    Main rules:
    - Draw all comments with one QWidget and QPainter.
    - Use fixed-speed movement.
    - Split overlay area into lanes.
    - Spawn a comment only when a lane has enough space.
    - If no lane is available, keep the comment in the queue.
    """

    def __init__(
        self,
        settings: AppSettings | None = None,
        enable_dummy_spawner: bool = False,
    ) -> None:
        super().__init__()

        self.settings = settings or AppSettings()

        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.WindowTransparentForInput
            | Qt.Tool
        )

        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)

        screen_geometry = QApplication.primaryScreen().geometry()
        self.screen_width = screen_geometry.width()
        self.screen_height = screen_geometry.height()
        self.setGeometry(0, 0, self.screen_width, self.screen_height)

        self.font = QFont(
            self.settings.font_family,
            self.settings.font_size,
            QFont.Bold,
        )
        self.font_metrics = QFontMetrics(self.font)

        self.active_comments: list[MovingComment] = []
        self.pending_comments: Deque[str] = deque()
        self.last_spawned_lane_index: int | None = None

        self.overlay_top = int(self.screen_height *
                               self.settings.overlay_top_ratio)
        self.overlay_bottom = int(
            self.screen_height * self.settings.overlay_bottom_ratio)

        self.lane_height = max(
            self.settings.lane_height_px,
            self.font_metrics.height() + self.settings.lane_vertical_padding_px,
        )

        self.lane_y_positions = self._build_lanes()

        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._animation_tick)
        self.animation_timer.start(self.settings.animation_interval_ms)

        self.spawn_timer = QTimer(self)
        self.spawn_timer.setSingleShot(True)
        self.spawn_timer.timeout.connect(self._on_spawn_timer_timeout)
        self._schedule_next_spawn()

        self.stream_spawn_timer = QTimer(self)
        self.stream_spawn_timer.setSingleShot(True)
        self.stream_spawn_timer.timeout.connect(
            self._on_stream_spawn_timer_timeout)

        self.dummy_timer: QTimer | None = None
        if enable_dummy_spawner:
            self.dummy_timer = QTimer(self)
            self.dummy_timer.timeout.connect(self._enqueue_dummy_comment)
            self.dummy_timer.start(1500)

        self.dummy_comments = [
            "かわいい",
            "やったぜ",
            "仲間！",
            "これからどんな冒険が待ってるんだろうか",
            "てぇてぇ",
            "冒険始まる！",
            "神ゲー",
            "相棒との出会い、神",
            "いいなあ",
            "癒し",
            "相棒",
            "最高！",
        ]

    def add_comment_batch(self, batch: CommentBatch) -> None:
        """
        Replace old pending comments with the newest API result.

        This prevents outdated comments from appearing after the scene/context
        has already changed.
        """

        old_pending_count = len(self.pending_comments)
        old_active_count = len(self.active_comments)

        self.pending_comments.clear()

        clear_active = getattr(
            self.settings,
            "clear_active_comments_on_new_batch",
            False,
        )

        if clear_active:
            self.active_comments.clear()
            self.update()

        comments = [*batch.comments, *batch.long_comments]

        for comment in comments:
            clean = str(comment).strip()
            if clean:
                self.pending_comments.append(clean)

        self._trim_pending_queue()

        print(
            "[overlay] new batch received: "
            f"cleared_pending={old_pending_count}, "
            f"cleared_active={old_active_count if clear_active else 0}, "
            f"queued_new={len(self.pending_comments)}"
        )

        self._try_spawn_from_queue()

    def add_comment(self, text: str) -> None:
        clean = str(text).strip()
        if clean:
            self.pending_comments.append(clean)

        self._trim_pending_queue()
        self._try_spawn_from_queue()

    def add_streamed_comment(self, text: str) -> None:
        clean = str(text).strip()

        if not clean:
            return

        self.pending_comments.append(clean)
        self._trim_pending_queue()

        if self.stream_spawn_timer.isActive():
            return

        self._try_spawn_from_queue()
        self._schedule_next_stream_spawn()

    def _build_lanes(self) -> list[int]:
        available_height = max(0, self.overlay_bottom - self.overlay_top)

        if available_height < self.lane_height:
            return [self.overlay_top + self.font_metrics.ascent()]

        lane_count = available_height // self.lane_height

        positions = []
        for lane_index in range(lane_count):
            baseline_y = (
                self.overlay_top
                + lane_index * self.lane_height
                + self.font_metrics.ascent()
            )
            positions.append(baseline_y)

        return positions

    def _enqueue_dummy_comment(self) -> None:
        self.pending_comments.append(random.choice(self.dummy_comments))
        self._trim_pending_queue()

    def _trim_pending_queue(self) -> None:
        max_queue_size = self.settings.max_pending_comments

        while len(self.pending_comments) > max_queue_size:
            self.pending_comments.popleft()

    def _try_spawn_from_queue(self) -> None:
        if not self.pending_comments:
            return

        if len(self.active_comments) >= self.settings.max_simultaneous_comments:
            return

        text = self.pending_comments[0]
        width = self.font_metrics.horizontalAdvance(text)

        lane_index = self._find_available_lane(width)

        if lane_index is None:
            return

        self.pending_comments.popleft()
        self._spawn_comment_now(text, width, lane_index)

    def _on_spawn_timer_timeout(self) -> None:
        self._try_spawn_from_queue()
        self._schedule_next_spawn()

    def _on_stream_spawn_timer_timeout(self) -> None:
        self._try_spawn_from_queue()

        if self.pending_comments:
            self._schedule_next_stream_spawn()

    def _schedule_next_spawn(self) -> None:
        min_ms = self.settings.comment_spawn_min_interval_ms
        max_ms = self.settings.comment_spawn_max_interval_ms

        if max_ms < min_ms:
            max_ms = min_ms

        interval_ms = random.randint(min_ms, max_ms)
        self.spawn_timer.start(interval_ms)

    def _schedule_next_stream_spawn(self) -> None:
        min_ms = self.settings.stream_comment_spawn_min_interval_ms
        max_ms = self.settings.stream_comment_spawn_max_interval_ms

        if max_ms < min_ms:
            max_ms = min_ms

        interval_ms = random.randint(min_ms, max_ms)
        self.stream_spawn_timer.start(interval_ms)

    def _find_available_lane(self, new_comment_width: int) -> int | None:
        """
        Return a randomized available lane.

        Rules:
        - Prefer empty lanes.
        - Otherwise use lanes where the previous comment has moved far enough.
        - Avoid using the same lane repeatedly when alternatives exist.
        """

        lane_indices = list(range(len(self.lane_y_positions)))
        random.shuffle(lane_indices)

        empty_lanes: list[int] = []
        available_lanes: list[int] = []

        for lane_index in lane_indices:
            comments_in_lane = [
                comment
                for comment in self.active_comments
                if comment.lane_index == lane_index
            ]

            if not comments_in_lane:
                empty_lanes.append(lane_index)
                continue

            rightmost_x = max(
                comment.x + comment.width for comment in comments_in_lane)
            free_space = self.screen_width - rightmost_x

            if free_space >= self.settings.min_comment_gap_px:
                available_lanes.append(lane_index)

        if empty_lanes:
            return self._choose_lane_randomly(empty_lanes)

        if available_lanes:
            return self._choose_lane_randomly(available_lanes)

        return None

    def _choose_lane_randomly(self, lanes: list[int]) -> int:
        """
        Pick a lane randomly, while avoiding the exact same lane twice in a row
        when possible.
        """

        if len(lanes) > 1 and self.last_spawned_lane_index in lanes:
            lanes = [
                lane
                for lane in lanes
                if lane != self.last_spawned_lane_index
            ]

        chosen = random.choice(lanes)
        self.last_spawned_lane_index = chosen
        return chosen

    def _spawn_comment_now(self, text: str, width: int, lane_index: int) -> None:
        comment = MovingComment(
            text=text,
            x=float(self.screen_width),
            y=float(self.lane_y_positions[lane_index]),
            speed=float(self.settings.comment_speed_px_per_tick),
            width=width,
            lane_index=lane_index,
        )

        self.active_comments.append(comment)

    def _animation_tick(self) -> None:
        next_active: list[MovingComment] = []

        for comment in self.active_comments:
            comment.x -= comment.speed

            if comment.x >= -comment.width:
                next_active.append(comment)

        self.active_comments = next_active
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.TextAntialiasing)
        painter.setFont(self.font)

        for comment in self.active_comments:
            x = int(comment.x)
            y = int(comment.y)

            # Lightweight outline. Cheaper than QGraphicsDropShadowEffect.
            painter.setPen(QColor(0, 0, 0, 220))
            painter.drawText(x + 2, y + 2, comment.text)
            painter.drawText(x - 1, y, comment.text)
            painter.drawText(x + 1, y, comment.text)
            painter.drawText(x, y - 1, comment.text)
            painter.drawText(x, y + 1, comment.text)

            painter.setPen(QColor(255, 255, 255, 255))
            painter.drawText(x, y, comment.text)


def main() -> None:
    app = QApplication(sys.argv)
    window = OverlayWindow(enable_dummy_spawner=True)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
