from __future__ import annotations

from PyQt5.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont, QFontMetrics, QGuiApplication, QPainter, QPen
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from danmaku.capture.capture_service import list_window_titles

from danmaku.models import AppSettings


class CaptureRegionSelector(QWidget):
    """Transparent full-screen overlay for selecting a capture rectangle."""

    selection_requested = pyqtSignal(tuple)
    selection_cancelled = pyqtSignal()

    def __init__(self) -> None:
        super().__init__(flags=Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowModality(Qt.ApplicationModal)
        self.setCursor(Qt.CrossCursor)
        self.setFocusPolicy(Qt.StrongFocus)

        self.start_position: QPoint | None = None
        self.selection_rect = QRect()

        virtual_geometry = self._build_virtual_geometry()
        self.setGeometry(virtual_geometry)
        self.show()
        self.activateWindow()

    def _build_virtual_geometry(self) -> QRect:
        screens = QGuiApplication.screens()
        if not screens:
            return QRect(0, 0, 0, 0)

        union_rect = screens[0].geometry()
        for screen in screens[1:]:
            union_rect = union_rect.united(screen.geometry())

        return union_rect

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return

        self.start_position = event.globalPos()
        self.selection_rect = QRect(self.start_position, self.start_position)
        self.update()

    def mouseMoveEvent(self, event) -> None:
        if self.start_position is None:
            return

        self.selection_rect = QRect(self.start_position, event.globalPos()).normalized()
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.LeftButton or self.start_position is None:
            return

        self.selection_rect = QRect(self.start_position, event.globalPos()).normalized()
        self.start_position = None

        if (
            self.selection_rect.width() < 100
            or self.selection_rect.height() < 100
        ):
            QMessageBox.warning(
                self,
                "Selection too small",
                "Capture region must be at least 100x100 pixels.",
            )
            self.selection_rect = QRect()
            self.update()
            return

        self.selection_requested.emit(
            self._logical_rect_to_physical(self.selection_rect)
        )
        self.close()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.selection_cancelled.emit()
            self.close()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 120))

        if self.selection_rect.isNull():
            return

        local_rect = QRect(
            self.selection_rect.topLeft() - self.geometry().topLeft(),
            self.selection_rect.size(),
        )

        painter.setBrush(QColor(0, 120, 215, 80))
        painter.setPen(QPen(QColor(0, 180, 255), 2))
        painter.drawRect(local_rect)

        text = f"{self.selection_rect.width()} x {self.selection_rect.height()}"
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(local_rect.topLeft() + QPoint(6, -8), text)

    def _logical_rect_to_physical(self, rect: QRect) -> tuple[int, int, int, int]:
        screen = QGuiApplication.screenAt(rect.topLeft()) or QGuiApplication.primaryScreen()
        ratio = screen.devicePixelRatio()

        left = int(round(rect.left() * ratio))
        top = int(round(rect.top() * ratio))
        width = int(round(rect.width() * ratio))
        height = int(round(rect.height() * ratio))

        return left, top, width, height


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

        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setPlaceholderText("GEMINI_API_KEY")
        self.api_key_input.setText(settings.api_key)

        self.model_input = QLineEdit()
        self.model_input.setText(settings.model_name)

        self.dummy_checkbox = QCheckBox("Use dummy API responses")
        self.dummy_checkbox.setChecked(settings.use_dummy_api)

        self.interval_input = QSpinBox()
        self.interval_input.setRange(2, 60)
        self.interval_input.setValue(settings.capture_interval_seconds)
        self.interval_input.setSuffix(" sec")

        self.window_selector = QComboBox()
        self.refresh_windows_button = QPushButton("Refresh windows")
        self._load_window_titles()

        self.capture_mode_label = QLabel(self._format_capture_mode_text())
        self.capture_region_label = QLabel(self._format_capture_region_text())
        self.select_capture_region_button = QPushButton("Select Capture Region")
        self.select_capture_region_button.clicked.connect(
            self._on_select_capture_region_clicked
        )

        self.font_size_input = QSpinBox()
        self.font_size_input.setRange(12, 96)
        self.font_size_input.setValue(settings.font_size)

        form = QFormLayout()
        form.addRow("Gemini API key", self.api_key_input)
        form.addRow("Model", self.model_input)
        form.addRow("Capture interval", self.interval_input)
        form.addRow("Overlay font size", self.font_size_input)
        form.addRow("", self.dummy_checkbox)
        form.addRow("Capture window", self.window_selector)
        form.addRow("", self.refresh_windows_button)
        form.addRow("Capture mode", self.capture_mode_label)
        form.addRow("Capture region", self.capture_region_label)
        form.addRow("", self.select_capture_region_button)
        self.refresh_windows_button.clicked.connect(self._load_window_titles)
        self.window_selector.currentIndexChanged.connect(
            self._on_window_selector_changed
        )

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
        self.settings.api_key = self.api_key_input.text().strip()
        self.settings.model_name = self.model_input.text().strip()
        self.settings.use_dummy_api = self.dummy_checkbox.isChecked()
        self.settings.capture_interval_seconds = self.interval_input.value()
        self.settings.font_size = self.font_size_input.value()
        self.settings.target_window_title = self.window_selector.currentData() or ""

        if self.settings.capture_mode != "region":
            self.settings.capture_mode = (
                "window"
                if self.settings.target_window_title
                else "full_screen"
            )

        self._refresh_capture_info()

    def set_running(self, is_running: bool) -> None:
        self.status_label.setText(
            "Status: running" if is_running else "Status: stopped")

    def _on_select_capture_region_clicked(self) -> None:
        self.region_selector = CaptureRegionSelector()
        self.region_selector.selection_requested.connect(
            self._on_capture_region_selected
        )
        self.region_selector.selection_cancelled.connect(
            self._on_capture_region_selection_cancelled
        )

    def _on_capture_region_selected(
        self,
        region: tuple[int, int, int, int],
    ) -> None:
        self.settings.capture_mode = "region"
        self.settings.capture_region = region
        self.settings.target_window_title = ""
        self.window_selector.setCurrentIndex(0)
        self._refresh_capture_info()

    def _on_capture_region_selection_cancelled(self) -> None:
        self._refresh_capture_info()

    def _on_window_selector_changed(self, index: int) -> None:
        if self.window_selector.currentData():
            self.settings.capture_mode = "window"
        else:
            self.settings.capture_mode = "full_screen"

        self._refresh_capture_info()

    def _refresh_capture_info(self) -> None:
        self.capture_mode_label.setText(self._format_capture_mode_text())
        self.capture_region_label.setText(self._format_capture_region_text())

    def _format_capture_mode_text(self) -> str:
        return self.settings.capture_mode.replace("_", " ").title()

    def _format_capture_region_text(self) -> str:
        if self.settings.capture_mode != "region":
            return "None"

        left, top, width, height = self.settings.capture_region
        if width <= 0 or height <= 0:
            return "Not selected"

        return f"x={left}, y={top}, width={width}, height={height}"

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


def main() -> None:
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = SettingsWindow(AppSettings())
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
