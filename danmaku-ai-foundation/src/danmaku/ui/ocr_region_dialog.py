from __future__ import annotations

from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QImage, QPainter, QPen
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class OcrRegionCanvas(QWidget):
    def __init__(
        self,
        image: QImage,
        region: tuple[float, float, float, float],
    ) -> None:
        super().__init__()
        self.image = image
        self.region = region
        self.drag_start: QPointF | None = None
        self.drag_end: QPointF | None = None
        self.setMinimumSize(720, 420)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)

    def _image_rect(self) -> QRectF:
        if self.image.isNull():
            return QRectF()
        available = QRectF(self.rect())
        scale = min(
            available.width() / self.image.width(),
            available.height() / self.image.height(),
        )
        width = self.image.width() * scale
        height = self.image.height() * scale
        return QRectF(
            (available.width() - width) / 2,
            (available.height() - height) / 2,
            width,
            height,
        )

    def _region_rect(self) -> QRectF:
        image_rect = self._image_rect()
        x, y, width, height = self.region
        return QRectF(
            image_rect.left() + x * image_rect.width(),
            image_rect.top() + y * image_rect.height(),
            width * image_rect.width(),
            height * image_rect.height(),
        )

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#050A13"))
        image_rect = self._image_rect()
        painter.drawImage(image_rect, self.image)

        selection = self._region_rect()
        painter.fillRect(selection, QColor(37, 99, 235, 60))
        painter.setPen(QPen(QColor("#60A5FA"), 3))
        painter.drawRect(selection)

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        point = QPointF(event.pos())
        if self._image_rect().contains(point):
            self.drag_start = point
            self.drag_end = point

    def mouseMoveEvent(self, event) -> None:
        if self.drag_start is None:
            return
        self.drag_end = self._clamp_to_image(QPointF(event.pos()))
        self._update_region_from_drag()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.LeftButton or self.drag_start is None:
            return
        self.drag_end = self._clamp_to_image(QPointF(event.pos()))
        self._update_region_from_drag()
        self.drag_start = None
        self.drag_end = None

    def _clamp_to_image(self, point: QPointF) -> QPointF:
        bounds = self._image_rect()
        return QPointF(
            min(bounds.right(), max(bounds.left(), point.x())),
            min(bounds.bottom(), max(bounds.top(), point.y())),
        )

    def _update_region_from_drag(self) -> None:
        if self.drag_start is None or self.drag_end is None:
            return
        bounds = self._image_rect()
        drag = QRectF(self.drag_start, self.drag_end).normalized()
        drag = drag.intersected(bounds)
        if drag.width() < 4 or drag.height() < 4:
            return
        self.region = (
            (drag.left() - bounds.left()) / bounds.width(),
            (drag.top() - bounds.top()) / bounds.height(),
            drag.width() / bounds.width(),
            drag.height() / bounds.height(),
        )
        self.update()


class OcrRegionDialog(QDialog):
    def __init__(
        self,
        image: QImage,
        region: tuple[float, float, float, float],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select OCR area")
        self.resize(920, 650)

        help_label = QLabel(
            "Drag over the part of the captured window that normally contains "
            "subtitles or dialogue. The area scales with the window."
        )
        help_label.setWordWrap(True)

        self.canvas = OcrRegionCanvas(image, region)
        full_image_button = QPushButton("Use full image")
        full_image_button.clicked.connect(self._use_full_image)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(help_label)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(full_image_button)
        layout.addWidget(buttons)

    @property
    def selected_region(self) -> tuple[float, float, float, float]:
        return self.canvas.region

    def _use_full_image(self) -> None:
        self.canvas.region = (0.0, 0.0, 1.0, 1.0)
        self.canvas.update()


def pil_image_to_qimage(image) -> QImage:
    rgb = image.convert("RGB")
    data = rgb.tobytes("raw", "RGB")
    return QImage(
        data,
        rgb.width,
        rgb.height,
        rgb.width * 3,
        QImage.Format_RGB888,
    ).copy()
