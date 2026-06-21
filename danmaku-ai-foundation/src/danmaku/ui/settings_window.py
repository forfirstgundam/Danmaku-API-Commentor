from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QButtonGroup,
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
    QRadioButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from danmaku.capture.capture_service import list_window_titles
from danmaku.models import AppSettings


APP_STYLE = """
QWidget {
    background-color: #111827;
    color: #E5E7EB;
    font-family: "Segoe UI";
    font-size: 20px;
}

QLabel#TitleLabel {
    color: #F9FAFB;
    font-size: 32px;
    font-weight: 700;
}

QLabel#StatusLabel {
    color: #93C5FD;
    font-weight: 600;
}

QTabWidget::pane {
    border: 1px solid #374151;
    border-radius: 12px;
    background-color: #111827;
}

QTabBar::tab {
    background-color: #1F2937;
    color: #D1D5DB;
    padding: 10px 18px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background-color: #2563EB;
    color: white;
}

QGroupBox {
    border: 1px solid #374151;
    border-radius: 14px;
    margin-top: 18px;
    padding: 16px;
    background-color: #1F2937;
    color: #93C5FD;
    font-weight: 700;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 6px;
}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit {
    background-color: #111827;
    border: 1px solid #4B5563;
    border-radius: 8px;
    padding: 7px;
    color: #F9FAFB;
}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus {
    border: 1px solid #60A5FA;
}

QCheckBox, QRadioButton {
    color: #E5E7EB;
    spacing: 8px;
}

QPushButton {
    background-color: #2563EB;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 22px;
    font-weight: 700;
}

QPushButton:hover {
    background-color: #1D4ED8;
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
    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()

    def __init__(self, settings: AppSettings) -> None:
        super().__init__()
        self.settings = settings

        self.setWindowTitle("Danmaku AI Settings")
        self.setMinimumWidth(760)
        self.setMinimumHeight(700)
        self.setStyleSheet(APP_STYLE)

        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout()
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(16)

        title = QLabel("Danmaku AI")
        title.setObjectName("TitleLabel")

        self.status_label = QLabel("Status: stopped")
        self.status_label.setObjectName("StatusLabel")

        header = QHBoxLayout()
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.status_label)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_api_tab(), "API")
        self.tabs.addTab(self._build_capture_tab(), "Capture")
        self.tabs.addTab(self._build_overlay_tab(), "Overlay")
        self.tabs.addTab(self._build_prompt_tab(), "Prompt")
        self.tabs.addTab(self._build_log_tab(), "Logging")

        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("StopButton")

        button_row = QHBoxLayout()
        button_row.addStretch()
        button_row.addWidget(self.stop_button)
        button_row.addWidget(self.start_button)

        root.addLayout(header)
        root.addWidget(self.tabs)
        root.addLayout(button_row)
        self.setLayout(root)

        self.start_button.clicked.connect(self._on_start_clicked)
        self.stop_button.clicked.connect(self._on_stop_clicked)

    def _build_api_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        group = QGroupBox("API Settings")
        form = QFormLayout()
        form.setHorizontalSpacing(24)
        form.setVerticalSpacing(12)

        self.provider_input = QComboBox()
        self.provider_input.addItem("Gemini", "gemini")
        self.provider_input.addItem("OpenAI", "openai")
        self._set_combo_data(self.provider_input, self.settings.api_provider)
        self.provider_input.currentIndexChanged.connect(self._update_api_key_placeholder)

        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setText(self.settings.api_key)
        self._update_api_key_placeholder()

        self.model_input = QLineEdit(self.settings.model_name)
        self.fallback_model_input = QLineEdit(self.settings.fallback_model_name)

        self.dummy_checkbox = QCheckBox("Use dummy API responses")
        self.dummy_checkbox.setChecked(self.settings.use_dummy_api)

        self.send_screenshot_checkbox = QCheckBox("Send screenshot to API")
        self.send_screenshot_checkbox.setChecked(self.settings.send_screenshot_to_api)

        self.streaming_checkbox = QCheckBox("Stream comments as they arrive")
        self.streaming_checkbox.setChecked(self.settings.use_streaming_api)

        self.api_image_size_input = QSpinBox()
        self.api_image_size_input.setRange(320, 1920)
        self.api_image_size_input.setSingleStep(64)
        self.api_image_size_input.setSuffix(" px")
        self.api_image_size_input.setValue(self.settings.api_image_max_dimension)

        self.api_image_quality_input = QSpinBox()
        self.api_image_quality_input.setRange(20, 95)
        self.api_image_quality_input.setSingleStep(5)
        self.api_image_quality_input.setSuffix(" quality")
        self.api_image_quality_input.setValue(self.settings.api_image_jpeg_quality)

        self.max_output_tokens_input = QSpinBox()
        self.max_output_tokens_input.setRange(128, 4096)
        self.max_output_tokens_input.setSingleStep(128)
        self.max_output_tokens_input.setSuffix(" tokens")
        self.max_output_tokens_input.setValue(self.settings.api_max_output_tokens)

        form.addRow("API provider", self.provider_input)
        form.addRow("API key", self.api_key_input)
        form.addRow("Model", self.model_input)
        form.addRow("Fallback model", self.fallback_model_input)
        form.addRow("API image max size", self.api_image_size_input)
        form.addRow("API JPEG quality", self.api_image_quality_input)
        form.addRow("Max output", self.max_output_tokens_input)
        form.addRow("", self.dummy_checkbox)
        form.addRow("", self.send_screenshot_checkbox)
        form.addRow("", self.streaming_checkbox)

        group.setLayout(form)
        layout.addWidget(group)
        layout.addStretch()
        return tab

    def _build_capture_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        group = QGroupBox("Capture Settings")
        form = QFormLayout()
        form.setHorizontalSpacing(24)
        form.setVerticalSpacing(12)

        self.interval_input = QSpinBox()
        self.interval_input.setRange(2, 60)
        self.interval_input.setSuffix(" sec")
        self.interval_input.setValue(self.settings.capture_interval_seconds)

        self.window_selector = QComboBox()
        self.refresh_windows_button = QPushButton("Refresh windows")
        self.refresh_windows_button.setObjectName("RefreshButton")
        self.refresh_windows_button.clicked.connect(self._load_window_titles)

        self._load_window_titles()

        form.addRow("Capture interval", self.interval_input)
        form.addRow("Capture window", self.window_selector)
        form.addRow("", self.refresh_windows_button)

        group.setLayout(form)
        layout.addWidget(group)
        layout.addStretch()
        return tab

    def _build_overlay_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        text_group = QGroupBox("Text Style")
        text_form = QFormLayout()
        text_form.setHorizontalSpacing(24)
        text_form.setVerticalSpacing(12)

        self.font_family_input = QLineEdit(self.settings.font_family)

        self.font_size_input = QSpinBox()
        self.font_size_input.setRange(12, 96)
        self.font_size_input.setValue(self.settings.font_size)

        text_form.addRow("Font family", self.font_family_input)
        text_form.addRow("Font size", self.font_size_input)
        text_group.setLayout(text_form)

        area_group = QGroupBox("Overlay Area")
        area_form = QFormLayout()
        area_form.setHorizontalSpacing(24)
        area_form.setVerticalSpacing(12)

        self.overlay_top_input = QDoubleSpinBox()
        self.overlay_top_input.setRange(0.0, 1.0)
        self.overlay_top_input.setSingleStep(0.01)
        self.overlay_top_input.setDecimals(2)
        self.overlay_top_input.setValue(self.settings.overlay_top_ratio)

        self.overlay_bottom_input = QDoubleSpinBox()
        self.overlay_bottom_input.setRange(0.0, 1.0)
        self.overlay_bottom_input.setSingleStep(0.01)
        self.overlay_bottom_input.setDecimals(2)
        self.overlay_bottom_input.setValue(self.settings.overlay_bottom_ratio)

        area_form.addRow("Top ratio", self.overlay_top_input)
        area_form.addRow("Bottom ratio", self.overlay_bottom_input)
        area_group.setLayout(area_form)

        lane_group = QGroupBox("Comment Flow")
        lane_form = QFormLayout()
        lane_form.setHorizontalSpacing(24)
        lane_form.setVerticalSpacing(12)

        self.lane_height_input = QSpinBox()
        self.lane_height_input.setRange(10, 120)
        self.lane_height_input.setSuffix(" px")
        self.lane_height_input.setValue(self.settings.lane_height_px)

        self.lane_padding_input = QSpinBox()
        self.lane_padding_input.setRange(0, 80)
        self.lane_padding_input.setSuffix(" px")
        self.lane_padding_input.setValue(self.settings.lane_vertical_padding_px)

        self.min_gap_input = QSpinBox()
        self.min_gap_input.setRange(0, 1000)
        self.min_gap_input.setSuffix(" px")
        self.min_gap_input.setValue(self.settings.min_comment_gap_px)

        self.max_simultaneous_input = QSpinBox()
        self.max_simultaneous_input.setRange(1, 200)
        self.max_simultaneous_input.setValue(self.settings.max_simultaneous_comments)

        self.max_pending_input = QSpinBox()
        self.max_pending_input.setRange(1, 500)
        self.max_pending_input.setValue(self.settings.max_pending_comments)

        self.animation_interval_input = QSpinBox()
        self.animation_interval_input.setRange(8, 100)
        self.animation_interval_input.setSuffix(" ms")
        self.animation_interval_input.setValue(self.settings.animation_interval_ms)

        self.spawn_min_input = QSpinBox()
        self.spawn_min_input.setRange(0, 10000)
        self.spawn_min_input.setSuffix(" ms")
        self.spawn_min_input.setValue(self.settings.comment_spawn_min_interval_ms)

        self.spawn_max_input = QSpinBox()
        self.spawn_max_input.setRange(0, 20000)
        self.spawn_max_input.setSuffix(" ms")
        self.spawn_max_input.setValue(self.settings.comment_spawn_max_interval_ms)

        self.speed_input = QDoubleSpinBox()
        self.speed_input.setRange(1.0, 100.0)
        self.speed_input.setSingleStep(1.0)
        self.speed_input.setDecimals(1)
        self.speed_input.setValue(self.settings.comment_speed_px_per_tick)

        self.clear_active_checkbox = QCheckBox("Clear active comments on new batch")
        self.clear_active_checkbox.setChecked(
            self.settings.clear_active_comments_on_new_batch
        )

        lane_form.addRow("Lane height", self.lane_height_input)
        lane_form.addRow("Lane padding", self.lane_padding_input)
        lane_form.addRow("Minimum gap", self.min_gap_input)
        lane_form.addRow("Max simultaneous", self.max_simultaneous_input)
        lane_form.addRow("Max pending", self.max_pending_input)
        lane_form.addRow("Animation interval", self.animation_interval_input)
        lane_form.addRow("Spawn min interval", self.spawn_min_input)
        lane_form.addRow("Spawn max interval", self.spawn_max_input)
        lane_form.addRow("Speed per tick", self.speed_input)
        lane_form.addRow("", self.clear_active_checkbox)

        lane_group.setLayout(lane_form)

        layout.addWidget(text_group)
        layout.addWidget(area_group)
        layout.addWidget(lane_group)
        layout.addStretch()
        return tab

    def _build_prompt_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        group = QGroupBox("Prompt Settings")
        form = QFormLayout()
        form.setHorizontalSpacing(24)
        form.setVerticalSpacing(12)

        self.default_prompt_radio = QRadioButton("Default system prompt")
        self.custom_prompt_radio = QRadioButton("User custom prompt")

        self.prompt_mode_group = QButtonGroup(self)
        self.prompt_mode_group.addButton(self.default_prompt_radio)
        self.prompt_mode_group.addButton(self.custom_prompt_radio)

        self.custom_prompt_input = QTextEdit()
        self.custom_prompt_input.setMinimumHeight(320)
        self.custom_prompt_input.setPlainText(self._read_default_prompt())

        self.default_prompt_radio.setChecked(True)
        prompt_mode_row = QHBoxLayout()
        prompt_mode_row.addWidget(self.default_prompt_radio)
        prompt_mode_row.addWidget(self.custom_prompt_radio)
        prompt_mode_row.addStretch()

        self.reset_prompt_button = QPushButton("Reset to current prompt")
        self.reset_prompt_button.clicked.connect(self._reset_custom_prompt_editor)

        self.default_prompt_radio.toggled.connect(self._update_prompt_editor_enabled)
        self.custom_prompt_radio.toggled.connect(self._update_prompt_editor_enabled)
        self._update_prompt_editor_enabled()

        form.addRow("Prompt mode", prompt_mode_row)
        form.addRow("Custom prompt", self.custom_prompt_input)
        form.addRow("", self.reset_prompt_button)

        group.setLayout(form)
        layout.addWidget(group)
        layout.addStretch()
        return tab

    def _build_log_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        group = QGroupBox("Logging")
        form = QFormLayout()
        form.setHorizontalSpacing(24)
        form.setVerticalSpacing(12)

        self.save_captures_checkbox = QCheckBox("Save captures")
        self.save_captures_checkbox.setChecked(self.settings.save_captures)

        self.save_comments_checkbox = QCheckBox("Save comments")
        self.save_comments_checkbox.setChecked(self.settings.save_comments)

        self.save_api_images_checkbox = QCheckBox("Save API images")
        self.save_api_images_checkbox.setChecked(self.settings.save_api_images)

        self.log_root_input = QLineEdit(str(self.settings.log_root_dir))

        form.addRow("", self.save_captures_checkbox)
        form.addRow("", self.save_comments_checkbox)
        form.addRow("", self.save_api_images_checkbox)
        form.addRow("Log root", self.log_root_input)

        group.setLayout(form)
        layout.addWidget(group)
        layout.addStretch()
        return tab

    def apply_to_settings(self) -> None:
        self.settings.api_provider = self.provider_input.currentData()
        self.settings.api_key = self.api_key_input.text().strip()
        self.settings.model_name = self.model_input.text().strip()
        self.settings.fallback_model_name = self.fallback_model_input.text().strip()
        self.settings.use_dummy_api = self.dummy_checkbox.isChecked()
        self.settings.send_screenshot_to_api = self.send_screenshot_checkbox.isChecked()
        self.settings.use_streaming_api = self.streaming_checkbox.isChecked()

        self.settings.api_image_max_dimension = self.api_image_size_input.value()
        self.settings.api_image_jpeg_quality = self.api_image_quality_input.value()
        self.settings.api_max_output_tokens = self.max_output_tokens_input.value()

        self.settings.capture_interval_seconds = self.interval_input.value()
        self.settings.target_window_title = self.window_selector.currentData() or ""

        self.settings.font_family = (
            self.font_family_input.text().strip() or "Malgun Gothic"
        )
        self.settings.font_size = self.font_size_input.value()
        self.settings.overlay_top_ratio = self.overlay_top_input.value()
        self.settings.overlay_bottom_ratio = self.overlay_bottom_input.value()

        self.settings.lane_height_px = self.lane_height_input.value()
        self.settings.lane_vertical_padding_px = self.lane_padding_input.value()
        self.settings.min_comment_gap_px = self.min_gap_input.value()
        self.settings.max_simultaneous_comments = self.max_simultaneous_input.value()
        self.settings.max_pending_comments = self.max_pending_input.value()
        self.settings.animation_interval_ms = self.animation_interval_input.value()
        self.settings.comment_spawn_min_interval_ms = self.spawn_min_input.value()
        self.settings.comment_spawn_max_interval_ms = self.spawn_max_input.value()
        self.settings.comment_speed_px_per_tick = self.speed_input.value()
        self.settings.clear_active_comments_on_new_batch = (
            self.clear_active_checkbox.isChecked()
        )

        self.settings.save_captures = self.save_captures_checkbox.isChecked()
        self.settings.save_comments = self.save_comments_checkbox.isChecked()
        self.settings.save_api_images = self.save_api_images_checkbox.isChecked()
        self.settings.log_root_dir = Path(self.log_root_input.text().strip() or "logs")

        self._apply_prompt_selection()

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
        if self.overlay_top_input.value() >= self.overlay_bottom_input.value():
            QMessageBox.warning(
                self,
                "Invalid overlay area",
                "Top ratio must be smaller than bottom ratio.",
            )
            return False

        if self.spawn_min_input.value() > self.spawn_max_input.value():
            QMessageBox.warning(
                self,
                "Invalid spawn interval",
                "Spawn min interval must be smaller than or equal to spawn max interval.",
            )
            return False

        if self.custom_prompt_radio.isChecked():
            prompt = self.custom_prompt_input.toPlainText().strip()
            if not prompt:
                QMessageBox.warning(
                    self,
                    "Invalid prompt",
                    "Custom prompt cannot be empty.",
                )
                return False

        return True

    def _load_window_titles(self) -> None:
        current = ""

        if hasattr(self, "window_selector"):
            current = self.window_selector.currentData() or self.window_selector.currentText()

        self.window_selector.clear()
        self.window_selector.addItem("Full screen", "")

        try:
            for title in list_window_titles():
                self.window_selector.addItem(title, title)
        except Exception as exc:
            print(f"[ui] failed to list windows: {exc}")

        if self.settings.target_window_title:
            current = self.settings.target_window_title

        if current:
            index = self.window_selector.findData(current)
            if index < 0:
                index = self.window_selector.findText(current)
            if index >= 0:
                self.window_selector.setCurrentIndex(index)

    def _update_api_key_placeholder(self) -> None:
        if not hasattr(self, "api_key_input"):
            return

        provider = self.provider_input.currentData()

        if provider == "openai":
            self.api_key_input.setPlaceholderText("OPENAI_API_KEY")
        else:
            self.api_key_input.setPlaceholderText("GEMINI_API_KEY")

    def _set_combo_data(self, combo: QComboBox, value: str) -> None:
        index = combo.findData(value)
        combo.setCurrentIndex(index if index >= 0 else 0)

    def _project_root(self) -> Path:
        return Path.cwd()

    def _prompt_path(self) -> Path:
        return self._project_root() / "prompts" / "system_prompt.txt"

    def _default_prompt_backup_path(self) -> Path:
        return self._project_root() / "prompts" / "system_prompt.default.txt"

    def _read_current_system_prompt(self) -> str:
        path = self._prompt_path()
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def _ensure_default_prompt_backup(self) -> None:
        prompt_path = self._prompt_path()
        backup_path = self._default_prompt_backup_path()

        if backup_path.exists():
            return

        backup_path.parent.mkdir(parents=True, exist_ok=True)

        if prompt_path.exists():
            backup_path.write_text(
                prompt_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        else:
            backup_path.write_text("", encoding="utf-8")

    def _read_default_prompt(self) -> str:
        self._ensure_default_prompt_backup()
        path = self._default_prompt_backup_path()
        return path.read_text(encoding="utf-8")

    def _apply_prompt_selection(self) -> None:
        prompt_path = self._prompt_path()
        prompt_path.parent.mkdir(parents=True, exist_ok=True)

        self._ensure_default_prompt_backup()

        if self.custom_prompt_radio.isChecked():
            prompt_text = self.custom_prompt_input.toPlainText().strip()
        else:
            prompt_text = self._read_default_prompt().strip()

        prompt_path.write_text(prompt_text, encoding="utf-8")

    def _reset_custom_prompt_editor(self) -> None:
        self.custom_prompt_input.setPlainText(self._read_current_system_prompt())

    def _update_prompt_editor_enabled(self) -> None:
        is_custom = self.custom_prompt_radio.isChecked()

        if is_custom:
            self.custom_prompt_input.setPlainText(self._read_current_system_prompt())
        else:
            self.custom_prompt_input.setPlainText(self._read_default_prompt())

        self.custom_prompt_input.setEnabled(is_custom)


def main() -> None:
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = SettingsWindow(AppSettings())
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()