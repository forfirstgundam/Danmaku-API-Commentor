from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from danmaku.capture.capture_service import list_window_titles
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from danmaku.models import AppSettings


class SettingsWindow(QWidget):
    """Minimal control window for the MVP."""

    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()

    def __init__(self, settings: AppSettings) -> None:
        super().__init__()
        self.settings = settings

        self.setWindowTitle("Danmaku AI Settings")
        self.setMinimumWidth(420)

        self.status_label = QLabel("Status: stopped")

        self.provider_input = QComboBox()
        self.provider_input.addItem("Gemini", "gemini")
        self.provider_input.addItem("OpenAI", "openai")
        provider_index = self.provider_input.findData(settings.api_provider)
        self.provider_input.setCurrentIndex(max(0, provider_index))
        self.provider_input.currentIndexChanged.connect(
            self._update_api_key_placeholder)

        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setText(settings.api_key)
        self._update_api_key_placeholder()

        self.model_input = QLineEdit()
        self.model_input.setText(settings.model_name)

        self.fallback_model_input = QLineEdit()
        self.fallback_model_input.setPlaceholderText("Optional fallback model")
        self.fallback_model_input.setText(settings.fallback_model_name)

        self.dummy_checkbox = QCheckBox("Use dummy API responses")
        self.dummy_checkbox.setChecked(settings.use_dummy_api)

        self.send_screenshot_checkbox = QCheckBox("Send screenshot to API")
        self.send_screenshot_checkbox.setChecked(settings.send_screenshot_to_api)

        self.streaming_checkbox = QCheckBox("Stream comments as they arrive")
        self.streaming_checkbox.setChecked(settings.use_streaming_api)

        self.interval_input = QSpinBox()
        self.interval_input.setRange(2, 60)
        self.interval_input.setValue(settings.capture_interval_seconds)
        self.interval_input.setSuffix(" sec")

        self.api_image_size_input = QSpinBox()
        self.api_image_size_input.setRange(320, 1920)
        self.api_image_size_input.setSingleStep(64)
        self.api_image_size_input.setValue(settings.api_image_max_dimension)
        self.api_image_size_input.setSuffix(" px")

        self.api_image_quality_input = QSpinBox()
        self.api_image_quality_input.setRange(20, 95)
        self.api_image_quality_input.setSingleStep(5)
        self.api_image_quality_input.setValue(settings.api_image_jpeg_quality)
        self.api_image_quality_input.setSuffix(" quality")

        self.max_output_tokens_input = QSpinBox()
        self.max_output_tokens_input.setRange(128, 2048)
        self.max_output_tokens_input.setSingleStep(128)
        self.max_output_tokens_input.setValue(settings.api_max_output_tokens)
        self.max_output_tokens_input.setSuffix(" tokens")

        self.window_selector = QComboBox()
        self.refresh_windows_button = QPushButton("Refresh windows")
        self._load_window_titles()

        self.font_size_input = QSpinBox()
        self.font_size_input.setRange(12, 96)
        self.font_size_input.setValue(settings.font_size)

        form = QFormLayout()
        form.addRow("API provider", self.provider_input)
        form.addRow("API key", self.api_key_input)
        form.addRow("Model", self.model_input)
        form.addRow("Fallback model", self.fallback_model_input)
        form.addRow("Capture interval", self.interval_input)
        form.addRow("API image max size", self.api_image_size_input)
        form.addRow("API JPEG quality", self.api_image_quality_input)
        form.addRow("Max output", self.max_output_tokens_input)
        form.addRow("Overlay font size", self.font_size_input)
        form.addRow("", self.dummy_checkbox)
        form.addRow("", self.send_screenshot_checkbox)
        form.addRow("", self.streaming_checkbox)
        form.addRow("Capture window", self.window_selector)
        form.addRow("", self.refresh_windows_button)
        self.refresh_windows_button.clicked.connect(self._load_window_titles)

        group = QGroupBox("Runtime settings")
        group.setLayout(form)

        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")

        button_row = QHBoxLayout()
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.stop_button)

        layout = QVBoxLayout()
        layout.addWidget(group)
        layout.addWidget(self.status_label)
        layout.addLayout(button_row)
        self.setLayout(layout)

        self.start_button.clicked.connect(self._on_start_clicked)
        self.stop_button.clicked.connect(self._on_stop_clicked)

    def apply_to_settings(self) -> None:
        self.settings.api_provider = self.provider_input.currentData()
        self.settings.api_key = self.api_key_input.text().strip()
        self.settings.model_name = self.model_input.text().strip()
        self.settings.fallback_model_name = self.fallback_model_input.text().strip()
        self.settings.use_dummy_api = self.dummy_checkbox.isChecked()
        self.settings.send_screenshot_to_api = self.send_screenshot_checkbox.isChecked()
        self.settings.use_streaming_api = self.streaming_checkbox.isChecked()
        self.settings.capture_interval_seconds = self.interval_input.value()
        self.settings.api_image_max_dimension = self.api_image_size_input.value()
        self.settings.api_image_jpeg_quality = self.api_image_quality_input.value()
        self.settings.api_max_output_tokens = self.max_output_tokens_input.value()
        self.settings.font_size = self.font_size_input.value()
        self.settings.target_window_title = self.window_selector.currentData() or ""

    def set_running(self, is_running: bool) -> None:
        self.status_label.setText(
            "Status: running" if is_running else "Status: stopped")

    def _on_start_clicked(self) -> None:
        self.apply_to_settings()
        self.set_running(True)
        self.start_requested.emit()

    def _on_stop_clicked(self) -> None:
        self.set_running(False)
        self.stop_requested.emit()

    def _load_window_titles(self) -> None:
        current = self.window_selector.currentText() if hasattr(
            self, "window_selector") else ""

        self.window_selector.clear()
        self.window_selector.addItem("Full screen", "")

        try:
            for title in list_window_titles():
                self.window_selector.addItem(title, title)
        except Exception as exc:
            print(f"[ui] failed to list windows: {exc}")

        if current:
            index = self.window_selector.findText(current)
            if index >= 0:
                self.window_selector.setCurrentIndex(index)

    def _update_api_key_placeholder(self) -> None:
        provider = self.provider_input.currentData()
        placeholder = "OPENAI_API_KEY" if provider == "openai" else "GEMINI_API_KEY"
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
