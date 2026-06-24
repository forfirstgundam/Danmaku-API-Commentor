from __future__ import annotations

import os
from pathlib import Path

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QScrollArea,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from danmaku.capture.capture_service import list_windows
from danmaku.config import (
    API_KEY_ENV_BY_PROVIDER,
    api_key_env_for_provider,
    default_fallback_model_for_provider,
    default_model_for_provider,
)
from danmaku.models import AppSettings, SessionProfile


APP_STYLE = """
QWidget {
    background-color: #111827;
    color: #E5E7EB;
    font-family: "Segoe UI";
    font-size: 16pt;
}

QLabel#TitleLabel {
    color: #F9FAFB;
    font-size: 25pt;
    font-weight: 700;
}

QLabel#StatusLabel {
    color: #93C5FD;
    font-size: 14pt;
    font-weight: 600;
}

/* Tabs */
QTabWidget::pane {
    border: 1px solid #374151;
    border-radius: 12px;
    background-color: #111827;
}

QTabBar::tab {
    background-color: #1F2937;
    color: #D1D5DB;
    min-width: 90px;
    min-height: 34px;
    padding: 10px 22px;
    margin-right: 3px;
    font-size: 14pt;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
}

QTabBar::tab:selected {
    background-color: #2563EB;
    color: white;
}

/* Section boxes */
QGroupBox {
    border: 1px solid #374151;
    border-radius: 14px;
    margin-top: 24px;
    padding: 22px;
    background-color: #1F2937;
    color: #93C5FD;
    font-size: 15pt;
    font-weight: 700;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 16px;
    padding: 0 8px;
}

/* Input controls */
QLineEdit,
QComboBox,
QSpinBox,
QDoubleSpinBox,
QTextEdit {
    background-color: #111827;
    border: 1px solid #4B5563;
    border-radius: 8px;
    min-height: 40px;
    padding: 7px 12px;
    color: #F9FAFB;
    font-size: 15pt;
}

QLineEdit:focus,
QComboBox:focus,
QSpinBox:focus,
QDoubleSpinBox:focus,
QTextEdit:focus {
    border: 2px solid #60A5FA;
}

/* Larger dropdown area */
QComboBox::drop-down {
    width: 34px;
    border: none;
}

/* Larger checkbox/radio text and click target */
QCheckBox,
QRadioButton {
    color: #E5E7EB;
    min-height: 34px;
    spacing: 12px;
    font-size: 14pt;
}

QCheckBox::indicator,
QRadioButton::indicator {
    width: 24px;
    height: 24px;
}

/* Buttons */
QPushButton {
    background-color: #2563EB;
    color: white;
    border: none;
    border-radius: 10px;
    min-height: 42px;
    padding: 7px 24px;
    font-size: 14pt;
    font-weight: 700;
}

QPushButton:hover {
    background-color: #1D4ED8;
}

QPushButton:disabled {
    background-color: #4B5563;
    color: #9CA3AF;
}

QPushButton#StopButton {
    background-color: #374151;
}

QPushButton#StopButton:hover {
    background-color: #4B5563;
}

QPushButton#RefreshButton {
    background-color: #0F766E;
}

QPushButton#RefreshButton:hover {
    background-color: #0D9488;
}
"""


class SettingsWindow(QWidget):
    """Tabbed runtime settings window."""

    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    session_profile_updated = pyqtSignal()

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
        self.tabs.addTab(self._build_context_tab(), "Context")
        self.tabs.addTab(self._build_capture_tab(), "Capture")
        self.tabs.addTab(self._build_logging_tab(), "Logging")
        self.tabs.addTab(
            self._make_scrollable(self._build_overlay_tab()),
            "Overlay",
        )

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
        for label, provider in (
            ("Gemini", "gemini"),
            ("OpenAI", "openai"),
            ("Anthropic", "anthropic"),
            ("DeepInfra", "deepinfra"),
            ("Together AI", "together"),
            ("Mistral", "mistral"),
            ("Groq", "groq"),
            ("xAI", "xai"),
        ):
            self.provider_input.addItem(label, provider)
        provider_index = self.provider_input.findData(
            self.settings.api_provider
        )
        self.provider_input.setCurrentIndex(max(0, provider_index))

        self._api_keys_by_provider = {
            provider: os.getenv(env_name, "").strip()
            for provider, env_name in API_KEY_ENV_BY_PROVIDER.items()
        }
        self._api_keys_by_provider[self.settings.api_provider] = (
            self.settings.api_key
        )
        self._fallback_models_by_provider = {
            provider: default_fallback_model_for_provider(provider)
            for provider in API_KEY_ENV_BY_PROVIDER
        }
        self._fallback_models_by_provider[self.settings.api_provider] = (
            self.settings.fallback_model_name
        )
        self._last_api_provider = self.settings.api_provider
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
            self._on_provider_changed
        )
        self._update_api_key_placeholder()
        return tab

    def _build_context_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        description_group = QGroupBox("Stream Description")
        description_layout = QVBoxLayout()

        description_help = QLabel(
            "Write a short, natural description such as “.hack//SIGN "
            "watching” or “Pokémon Black playthrough”. A new description "
            "is interpreted once when the first API request starts."
        )
        description_help.setWordWrap(True)

        self.stream_description_input = QTextEdit()
        self.stream_description_input.setMinimumHeight(120)
        self.stream_description_input.setPlaceholderText(
            ".hack//SIGN watching\nPokémon Black playthrough"
        )
        self.stream_description_input.setPlainText(
            self.settings.user_stream_description
        )

        description_layout.addWidget(description_help)
        description_layout.addWidget(self.stream_description_input)
        description_group.setLayout(description_layout)

        profile_group = QGroupBox("Interpreted Session Profile")
        profile_form = self._new_form()
        profile = self.settings.session_profile

        self.profile_title_input = QLineEdit(profile.title)

        self.profile_content_type_input = QComboBox()
        for label, value in (
            ("Unknown", "unknown"),
            ("Anime", "anime"),
            ("Game", "game"),
            ("Video", "video"),
            ("Manga", "manga"),
            ("Other", "other"),
        ):
            self.profile_content_type_input.addItem(label, value)
        type_index = self.profile_content_type_input.findData(
            profile.content_type
        )
        self.profile_content_type_input.setCurrentIndex(max(0, type_index))

        self.profile_activity_input = QLineEdit(profile.activity)
        self.profile_episode_input = QLineEdit(profile.episode)
        self.profile_episode_input.setPlaceholderText("Unknown")
        self.profile_subtitle_input = QLineEdit(profile.subtitle_language)
        self.profile_subtitle_input.setPlaceholderText("Unknown")

        self.profile_status_label = QLabel()
        self.profile_status_label.setWordWrap(True)
        self._show_profile_status(profile)

        self.apply_profile_button = QPushButton("Use edited profile")
        self.apply_profile_button.setObjectName("RefreshButton")
        self.apply_profile_button.clicked.connect(
            self._on_apply_profile_clicked
        )

        profile_form.addRow("Title", self.profile_title_input)
        profile_form.addRow(
            "Content type",
            self.profile_content_type_input,
        )
        profile_form.addRow("Activity", self.profile_activity_input)
        profile_form.addRow("Episode/chapter", self.profile_episode_input)
        profile_form.addRow(
            "Subtitle language",
            self.profile_subtitle_input,
        )
        profile_form.addRow("Status", self.profile_status_label)
        profile_form.addRow("", self.apply_profile_button)
        profile_group.setLayout(profile_form)

        layout.addWidget(description_group)
        layout.addWidget(profile_group)
        layout.addStretch()

        self.stream_description_input.textChanged.connect(
            self._on_stream_description_changed
        )
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

        self.capture_format_input = QComboBox()
        self.capture_format_input.addItem("JPEG", "JPEG")
        self.capture_format_input.addItem("PNG", "PNG")
        capture_format_index = self.capture_format_input.findData(
            self.settings.sample_capture_format.upper()
        )
        self.capture_format_input.setCurrentIndex(
            max(0, capture_format_index)
        )

        self.capture_quality_input = QSpinBox()
        self.capture_quality_input.setRange(20, 95)
        self.capture_quality_input.setSingleStep(5)
        self.capture_quality_input.setSuffix(" quality")
        self.capture_quality_input.setValue(
            self.settings.sample_capture_jpeg_quality
        )
        self.capture_format_input.currentIndexChanged.connect(
            self._update_capture_quality_enabled
        )
        self._update_capture_quality_enabled()

        self.api_image_size_input = QSpinBox()
        self.api_image_size_input.setRange(0, 1920)
        self.api_image_size_input.setSingleStep(64)
        self.api_image_size_input.setSuffix(" px")
        self.api_image_size_input.setSpecialValueText("Original")
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
        self.api_image_size_input.valueChanged.connect(
            self._update_api_image_quality_enabled
        )
        self.api_image_size_input.valueChanged.connect(
            self._sync_original_capture_format
        )
        self._update_api_image_quality_enabled(
            self.api_image_size_input.value()
        )
        self._sync_original_capture_format(
            self.api_image_size_input.value()
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

        image_form.addRow("Capture format", self.capture_format_input)
        image_form.addRow("Capture JPEG", self.capture_quality_input)
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

    def _update_api_image_quality_enabled(self, max_dimension: int) -> None:
        self.api_image_quality_input.setEnabled(max_dimension > 0)

    def _sync_original_capture_format(self, max_dimension: int) -> None:
        if max_dimension <= 0:
            png_index = self.capture_format_input.findData("PNG")
            self.capture_format_input.setCurrentIndex(png_index)
        self.capture_format_input.setEnabled(max_dimension > 0)
        self._update_capture_quality_enabled()

    def _update_capture_quality_enabled(self) -> None:
        self.capture_quality_input.setEnabled(
            self.capture_format_input.isEnabled()
            and self.capture_format_input.currentData() == "JPEG"
        )

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
        self.settings.user_stream_description = (
            self.stream_description_input.toPlainText().strip()
        )
        self._apply_profile_fields(mark_edited=False)

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
        self.settings.sample_capture_format = (
            "PNG"
            if self.api_image_size_input.value() <= 0
            else self.capture_format_input.currentData()
        )
        self.settings.sample_capture_jpeg_quality = (
            self.capture_quality_input.value()
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
    
    def _make_scrollable(self, content: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(content)
        return scroll

    def _warn(self, title: str, message: str) -> None:
        QMessageBox.warning(self, title, message)

    def _apply_profile_fields(self, *, mark_edited: bool) -> None:
        profile = self.settings.session_profile
        profile.title = self.profile_title_input.text().strip()
        profile.content_type = (
            self.profile_content_type_input.currentData() or "unknown"
        )
        profile.activity = self.profile_activity_input.text().strip()
        profile.episode = self.profile_episode_input.text().strip()
        profile.subtitle_language = (
            self.profile_subtitle_input.text().strip()
        )

        if mark_edited:
            description = (
                self.stream_description_input.toPlainText().strip()
            )
            self.settings.user_stream_description = description
            profile.source_description = description
            profile.status = "edited"
            profile.interpretation_error = ""
            self._show_profile_status(profile)

    def _on_apply_profile_clicked(self) -> None:
        self._apply_profile_fields(mark_edited=True)
        self.session_profile_updated.emit()

    def _on_stream_description_changed(self) -> None:
        description = self.stream_description_input.toPlainText().strip()
        if description != self.settings.session_profile.source_description:
            self.profile_status_label.setText(
                "Description changed; it will be interpreted on the first "
                "API request. Use ‘Use edited profile’ to keep the fields "
                "above instead."
            )

    def set_session_profile(self, profile: SessionProfile) -> None:
        """Display a profile produced by the background API worker."""
        self.profile_title_input.setText(profile.title)
        type_index = self.profile_content_type_input.findData(
            profile.content_type
        )
        self.profile_content_type_input.setCurrentIndex(max(0, type_index))
        self.profile_activity_input.setText(profile.activity)
        self.profile_episode_input.setText(profile.episode)
        self.profile_subtitle_input.setText(profile.subtitle_language)
        self._show_profile_status(profile)

    def _show_profile_status(self, profile: SessionProfile) -> None:
        labels = {
            "empty": "No stream description provided.",
            "pending": "Waiting for the first API request.",
            "interpreted": "Interpreted by the session model.",
            "fallback": "Using the raw description because interpretation failed.",
            "edited": "Using the profile edited by the user.",
        }
        message = labels.get(profile.status, profile.status or "Not generated")
        if profile.interpretation_error:
            message += f" Error: {profile.interpretation_error}"
        self.profile_status_label.setText(message)

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
        self.api_key_input.setPlaceholderText(
            api_key_env_for_provider(provider)
        )

    def _on_provider_changed(self) -> None:
        previous_provider = self._last_api_provider
        self._api_keys_by_provider[previous_provider] = (
            self.api_key_input.text().strip()
        )
        self._fallback_models_by_provider[previous_provider] = (
            self.fallback_model_input.text().strip()
        )

        provider = self.provider_input.currentData()
        previous_default = default_model_for_provider(previous_provider)
        if (
            not self.model_input.text().strip()
            or self.model_input.text().strip() == previous_default
        ):
            self.model_input.setText(default_model_for_provider(provider))

        self.api_key_input.setText(
            self._api_keys_by_provider.get(provider, "")
        )
        self.fallback_model_input.setText(
            self._fallback_models_by_provider.get(provider, "")
        )
        self._last_api_provider = provider
        self._update_api_key_placeholder()


def main() -> None:
    import sys

    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = SettingsWindow(AppSettings())
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
