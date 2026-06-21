from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from danmaku.capture.capture_service import list_windows
from danmaku.models import AppSettings


APP_STYLE = """
QWidget {
    background-color: #111827;
    color: #E5E7EB;
    font-family: "Segoe UI";
    font-size: 14px;
}
QLabel#TitleLabel {
    color: #F9FAFB;
    font-size: 26px;
    font-weight: 700;
}
QLabel#StatusLabel {
    color: #93C5FD;
    font-weight: 600;
}
QTabWidget::pane {
    border: 1px solid #374151;
    border-radius: 10px;
    background-color: #111827;
}
QTabBar::tab {
    background-color: #1F2937;
    color: #D1D5DB;
    padding: 9px 16px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background-color: #2563EB;
    color: white;
}
QGroupBox {
    border: 1px solid #374151;
    border-radius: 10px;
    margin-top: 16px;
    padding: 14px;
    background-color: #1F2937;
    color: #93C5FD;
    font-weight: 700;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 5px;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #111827;
    border: 1px solid #4B5563;
    border-radius: 6px;
    padding: 6px;
    color: #F9FAFB;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus,
QDoubleSpinBox:focus {
    border: 1px solid #60A5FA;
}
QCheckBox {
    color: #E5E7EB;
    spacing: 7px;
}
QPushButton {
    background-color: #2563EB;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 9px 18px;
    font-weight: 700;
}
QPushButton:hover {
    background-color: #1D4ED8;
}
QPushButton:disabled {
    background-color: #374151;
    color: #9CA3AF;
}
QPushButton#StopButton {
    background-color: #4B5563;
}
QPushButton#RefreshButton {
    background-color: #0F766E;
}
"""


class SettingsWindow(QWidget):
    """Tabbed runtime settings window."""

    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()

    def __init__(self, settings: AppSettings) -> None:
        super().__init__()
        self.settings = settings

        self.setWindowTitle("Danmaku AI Settings")
        self.setMinimumSize(760, 690)
        self.setStyleSheet(APP_STYLE)
        self._build_ui()
        self.set_running(False)

    def _build_ui(self) -> None:
        root = QVBoxLayout()
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        title = QLabel("Danmaku AI")
        title.setObjectName("TitleLabel")

        self.status_label = QLabel()
        self.status_label.setObjectName("StatusLabel")

        header = QHBoxLayout()
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.status_label)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_api_tab(), "API")
        self.tabs.addTab(self._build_capture_tab(), "Capture")
        self.tabs.addTab(self._build_overlay_tab(), "Overlay")
        self.tabs.addTab(self._build_logging_tab(), "Logging")

        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("StopButton")
        self.start_button = QPushButton("Start")

        buttons = QHBoxLayout()
        buttons.addStretch()
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.start_button)

        root.addLayout(header)
        root.addWidget(self.tabs)
        root.addLayout(buttons)
        self.setLayout(root)

        self.start_button.clicked.connect(self._on_start_clicked)
        self.stop_button.clicked.connect(self._on_stop_clicked)

    @staticmethod
    def _new_form() -> QFormLayout:
        form = QFormLayout()
        form.setHorizontalSpacing(24)
        form.setVerticalSpacing(10)
        return form

    def _build_api_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        group = QGroupBox("API Settings")
        form = self._new_form()

        self.provider_input = QComboBox()
        self.provider_input.addItem("Gemini", "gemini")
        self.provider_input.addItem("OpenAI", "openai")
        provider_index = self.provider_input.findData(
            self.settings.api_provider
        )
        self.provider_input.setCurrentIndex(max(0, provider_index))

        self.api_key_input = QLineEdit(self.settings.api_key)
        self.api_key_input.setEchoMode(QLineEdit.Password)

        self.model_input = QLineEdit(self.settings.model_name)
        self.fallback_model_input = QLineEdit(
            self.settings.fallback_model_name
        )
        self.fallback_model_input.setPlaceholderText(
            "Optional fallback model"
        )

        self.max_output_tokens_input = QSpinBox()
        self.max_output_tokens_input.setRange(128, 4096)
        self.max_output_tokens_input.setSingleStep(128)
        self.max_output_tokens_input.setSuffix(" tokens")
        self.max_output_tokens_input.setValue(
            self.settings.api_max_output_tokens
        )

        self.dummy_checkbox = QCheckBox("Use dummy API responses")
        self.dummy_checkbox.setChecked(self.settings.use_dummy_api)

        self.send_screenshot_checkbox = QCheckBox(
            "Send screenshots to API"
        )
        self.send_screenshot_checkbox.setChecked(
            self.settings.send_screenshot_to_api
        )

        self.streaming_checkbox = QCheckBox(
            "Stream comments as they arrive"
        )
        self.streaming_checkbox.setChecked(
            self.settings.use_streaming_api
        )

        form.addRow("API provider", self.provider_input)
        form.addRow("API key", self.api_key_input)
        form.addRow("Model", self.model_input)
        form.addRow("Fallback model", self.fallback_model_input)
        form.addRow("Max output", self.max_output_tokens_input)
        form.addRow("", self.dummy_checkbox)
        form.addRow("", self.send_screenshot_checkbox)
        form.addRow("", self.streaming_checkbox)

        group.setLayout(form)
        layout.addWidget(group)
        layout.addStretch()

        self.provider_input.currentIndexChanged.connect(
            self._update_api_key_placeholder
        )
        self._update_api_key_placeholder()
        return tab

    def _build_capture_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        timing_group = QGroupBox("Capture and Frame Sequence")
        timing_form = self._new_form()

        self.window_selector = QComboBox()
        self.refresh_windows_button = QPushButton("Refresh windows")
        self.refresh_windows_button.setObjectName("RefreshButton")
        self.refresh_windows_button.clicked.connect(self._load_windows)
        self._load_windows()

        self.interval_input = QSpinBox()
        self.interval_input.setRange(2, 60)
        self.interval_input.setSuffix(" sec")
        self.interval_input.setValue(
            self.settings.capture_interval_seconds
        )

        self.multi_frame_checkbox = QCheckBox(
            "Send recent frame sequence"
        )
        self.multi_frame_checkbox.setChecked(
            self.settings.use_multi_frame_context
        )

        self.sample_interval_input = QSpinBox()
        self.sample_interval_input.setRange(1, 10)
        self.sample_interval_input.setSuffix(" sec")
        self.sample_interval_input.setValue(
            self.settings.frame_sample_interval_seconds
        )

        self.frames_per_request_input = QSpinBox()
        self.frames_per_request_input.setRange(1, 8)
        self.frames_per_request_input.setSuffix(" frames")
        self.frames_per_request_input.setValue(
            self.settings.frames_per_request
        )

        timing_form.addRow("Capture window", self.window_selector)
        timing_form.addRow("", self.refresh_windows_button)
        timing_form.addRow("Comment/API interval", self.interval_input)
        timing_form.addRow("", self.multi_frame_checkbox)
        timing_form.addRow(
            "Frame sample interval",
            self.sample_interval_input,
        )
        timing_form.addRow(
            "Frames per request",
            self.frames_per_request_input,
        )
        timing_group.setLayout(timing_form)

        image_group = QGroupBox("API Image Compression")
        image_form = self._new_form()

        self.api_image_size_input = QSpinBox()
        self.api_image_size_input.setRange(320, 1920)
        self.api_image_size_input.setSingleStep(64)
        self.api_image_size_input.setSuffix(" px")
        self.api_image_size_input.setValue(
            self.settings.api_image_max_dimension
        )

        self.api_image_quality_input = QSpinBox()
        self.api_image_quality_input.setRange(20, 95)
        self.api_image_quality_input.setSingleStep(5)
        self.api_image_quality_input.setSuffix(" quality")
        self.api_image_quality_input.setValue(
            self.settings.api_image_jpeg_quality
        )

        self.history_image_size_input = QSpinBox()
        self.history_image_size_input.setRange(160, 1920)
        self.history_image_size_input.setSingleStep(32)
        self.history_image_size_input.setSuffix(" px")
        self.history_image_size_input.setValue(
            self.settings.history_image_max_dimension
        )

        self.history_image_quality_input = QSpinBox()
        self.history_image_quality_input.setRange(20, 95)
        self.history_image_quality_input.setSingleStep(5)
        self.history_image_quality_input.setSuffix(" quality")
        self.history_image_quality_input.setValue(
            self.settings.history_image_jpeg_quality
        )

        image_form.addRow("Current frame max size", self.api_image_size_input)
        image_form.addRow(
            "Current frame JPEG",
            self.api_image_quality_input,
        )
        image_form.addRow(
            "Historical frame max size",
            self.history_image_size_input,
        )
        image_form.addRow(
            "Historical frame JPEG",
            self.history_image_quality_input,
        )
        image_group.setLayout(image_form)

        layout.addWidget(timing_group)
        layout.addWidget(image_group)
        layout.addStretch()
        return tab

    def _build_overlay_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        style_group = QGroupBox("Text and Display Area")
        style_form = self._new_form()

        self.font_family_input = QLineEdit(self.settings.font_family)

        self.font_size_input = QSpinBox()
        self.font_size_input.setRange(12, 96)
        self.font_size_input.setValue(self.settings.font_size)

        self.overlay_top_input = QDoubleSpinBox()
        self.overlay_top_input.setRange(0.0, 1.0)
        self.overlay_top_input.setSingleStep(0.01)
        self.overlay_top_input.setDecimals(2)
        self.overlay_top_input.setValue(self.settings.overlay_top_ratio)

        self.overlay_bottom_input = QDoubleSpinBox()
        self.overlay_bottom_input.setRange(0.0, 1.0)
        self.overlay_bottom_input.setSingleStep(0.01)
        self.overlay_bottom_input.setDecimals(2)
        self.overlay_bottom_input.setValue(
            self.settings.overlay_bottom_ratio
        )

        style_form.addRow("Font family", self.font_family_input)
        style_form.addRow("Font size", self.font_size_input)
        style_form.addRow("Top ratio", self.overlay_top_input)
        style_form.addRow("Bottom ratio", self.overlay_bottom_input)
        style_group.setLayout(style_form)

        flow_group = QGroupBox("Comment Flow")
        flow_form = self._new_form()

        self.lane_height_input = self._pixel_spin(
            10,
            120,
            self.settings.lane_height_px,
        )
        self.lane_padding_input = self._pixel_spin(
            0,
            80,
            self.settings.lane_vertical_padding_px,
        )
        self.min_gap_input = self._pixel_spin(
            0,
            1000,
            self.settings.min_comment_gap_px,
        )

        self.max_simultaneous_input = QSpinBox()
        self.max_simultaneous_input.setRange(1, 200)
        self.max_simultaneous_input.setValue(
            self.settings.max_simultaneous_comments
        )

        self.max_pending_input = QSpinBox()
        self.max_pending_input.setRange(1, 500)
        self.max_pending_input.setValue(
            self.settings.max_pending_comments
        )

        self.animation_interval_input = self._millisecond_spin(
            8,
            100,
            self.settings.animation_interval_ms,
        )

        self.spawn_min_input = self._millisecond_spin(
            0,
            10000,
            self.settings.comment_spawn_min_interval_ms,
        )
        self.spawn_max_input = self._millisecond_spin(
            0,
            20000,
            self.settings.comment_spawn_max_interval_ms,
        )

        self.speed_input = QDoubleSpinBox()
        self.speed_input.setRange(1.0, 100.0)
        self.speed_input.setSingleStep(1.0)
        self.speed_input.setDecimals(1)
        self.speed_input.setValue(
            self.settings.comment_speed_px_per_tick
        )

        self.clear_active_checkbox = QCheckBox(
            "Clear active comments on each new batch"
        )
        self.clear_active_checkbox.setChecked(
            self.settings.clear_active_comments_on_new_batch
        )

        flow_form.addRow("Lane height", self.lane_height_input)
        flow_form.addRow("Lane padding", self.lane_padding_input)
        flow_form.addRow("Minimum gap", self.min_gap_input)
        flow_form.addRow(
            "Max simultaneous",
            self.max_simultaneous_input,
        )
        flow_form.addRow("Max pending", self.max_pending_input)
        flow_form.addRow(
            "Animation interval",
            self.animation_interval_input,
        )
        flow_form.addRow("Spawn min interval", self.spawn_min_input)
        flow_form.addRow("Spawn max interval", self.spawn_max_input)
        flow_form.addRow("Speed per tick", self.speed_input)
        flow_form.addRow("", self.clear_active_checkbox)
        flow_group.setLayout(flow_form)

        layout.addWidget(style_group)
        layout.addWidget(flow_group)
        layout.addStretch()
        return tab

    def _build_logging_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        group = QGroupBox("Run Logging")
        form = self._new_form()

        self.save_comments_checkbox = QCheckBox(
            "Save comments and request metadata"
        )
        self.save_comments_checkbox.setChecked(
            self.settings.save_comments
        )
        self.save_api_images_checkbox = QCheckBox(
            "Save compressed API images"
        )
        self.save_api_images_checkbox.setChecked(
            self.settings.save_api_images
        )
        self.log_root_input = QLineEdit(
            str(self.settings.log_root_dir)
        )

        form.addRow("", self.save_comments_checkbox)
        form.addRow("", self.save_api_images_checkbox)
        form.addRow("Log folder", self.log_root_input)
        group.setLayout(form)

        layout.addWidget(group)
        layout.addStretch()
        return tab

    @staticmethod
    def _pixel_spin(minimum: int, maximum: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSuffix(" px")
        spin.setValue(value)
        return spin

    @staticmethod
    def _millisecond_spin(
        minimum: int,
        maximum: int,
        value: int,
    ) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSuffix(" ms")
        spin.setValue(value)
        return spin

    def apply_to_settings(self) -> None:
        self.settings.api_provider = self.provider_input.currentData()
        self.settings.api_key = self.api_key_input.text().strip()
        self.settings.model_name = self.model_input.text().strip()
        self.settings.fallback_model_name = (
            self.fallback_model_input.text().strip()
        )
        self.settings.use_dummy_api = self.dummy_checkbox.isChecked()
        self.settings.send_screenshot_to_api = (
            self.send_screenshot_checkbox.isChecked()
        )
        self.settings.use_streaming_api = (
            self.streaming_checkbox.isChecked()
        )
        self.settings.api_max_output_tokens = (
            self.max_output_tokens_input.value()
        )

        self.settings.target_window_handle = int(
            self.window_selector.currentData() or 0
        )
        self.settings.target_window_title = (
            self.window_selector.currentText()
            if self.settings.target_window_handle
            else ""
        )
        self.settings.capture_interval_seconds = self.interval_input.value()
        self.settings.use_multi_frame_context = (
            self.multi_frame_checkbox.isChecked()
        )
        self.settings.frame_sample_interval_seconds = (
            self.sample_interval_input.value()
        )
        self.settings.frames_per_request = (
            self.frames_per_request_input.value()
        )
        self.settings.frame_buffer_size = max(
            self.settings.frame_buffer_size,
            self.settings.frames_per_request,
        )
        self.settings.api_image_max_dimension = (
            self.api_image_size_input.value()
        )
        self.settings.api_image_jpeg_quality = (
            self.api_image_quality_input.value()
        )
        self.settings.history_image_max_dimension = (
            self.history_image_size_input.value()
        )
        self.settings.history_image_jpeg_quality = (
            self.history_image_quality_input.value()
        )

        self.settings.font_family = (
            self.font_family_input.text().strip() or "Malgun Gothic"
        )
        self.settings.font_size = self.font_size_input.value()
        self.settings.overlay_top_ratio = self.overlay_top_input.value()
        self.settings.overlay_bottom_ratio = (
            self.overlay_bottom_input.value()
        )
        self.settings.lane_height_px = self.lane_height_input.value()
        self.settings.lane_vertical_padding_px = (
            self.lane_padding_input.value()
        )
        self.settings.min_comment_gap_px = self.min_gap_input.value()
        self.settings.max_simultaneous_comments = (
            self.max_simultaneous_input.value()
        )
        self.settings.max_pending_comments = self.max_pending_input.value()
        self.settings.animation_interval_ms = (
            self.animation_interval_input.value()
        )
        self.settings.comment_spawn_min_interval_ms = (
            self.spawn_min_input.value()
        )
        self.settings.comment_spawn_max_interval_ms = (
            self.spawn_max_input.value()
        )
        self.settings.comment_speed_px_per_tick = self.speed_input.value()
        self.settings.clear_active_comments_on_new_batch = (
            self.clear_active_checkbox.isChecked()
        )

        self.settings.save_comments = (
            self.save_comments_checkbox.isChecked()
        )
        self.settings.save_api_images = (
            self.save_api_images_checkbox.isChecked()
        )
        self.settings.log_root_dir = Path(
            self.log_root_input.text().strip() or "logs"
        )

    def set_running(self, is_running: bool) -> None:
        self.status_label.setText(
            "Status: running" if is_running else "Status: stopped"
        )
        self.start_button.setEnabled(not is_running)
        self.stop_button.setEnabled(is_running)

    def _on_start_clicked(self) -> None:
        if not self._validate_before_start():
            return
        self.apply_to_settings()
        self.set_running(True)
        self.start_requested.emit()

    def _on_stop_clicked(self) -> None:
        self.set_running(False)
        self.stop_requested.emit()

    def _validate_before_start(self) -> bool:
        if not self.model_input.text().strip():
            self._warn("Invalid model", "Model name cannot be empty.")
            return False

        if self.overlay_top_input.value() >= self.overlay_bottom_input.value():
            self._warn(
                "Invalid overlay area",
                "Top ratio must be smaller than bottom ratio.",
            )
            return False

        if self.spawn_min_input.value() > self.spawn_max_input.value():
            self._warn(
                "Invalid spawn interval",
                "Spawn minimum must not exceed spawn maximum.",
            )
            return False

        return True

    def _warn(self, title: str, message: str) -> None:
        QMessageBox.warning(self, title, message)

    def _load_windows(self) -> None:
        current_handle = (
            self.window_selector.currentData()
            if hasattr(self, "window_selector")
            else self.settings.target_window_handle
        )

        self.window_selector.clear()
        self.window_selector.addItem("Full screen", 0)

        try:
            for handle, title in list_windows():
                self.window_selector.addItem(title, handle)
        except Exception as exc:
            print(f"[ui] failed to list windows: {exc}")

        if current_handle:
            index = self.window_selector.findData(current_handle)
            if index >= 0:
                self.window_selector.setCurrentIndex(index)

    def _update_api_key_placeholder(self) -> None:
        provider = self.provider_input.currentData()
        placeholder = (
            "OPENAI_API_KEY"
            if provider == "openai"
            else "GEMINI_API_KEY"
        )
        self.api_key_input.setPlaceholderText(placeholder)


def main() -> None:
    import sys

    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = SettingsWindow(AppSettings())
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
