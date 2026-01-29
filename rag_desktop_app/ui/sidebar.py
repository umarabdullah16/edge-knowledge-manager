from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import QPropertyAnimation, QEasingCurve


class Sidebar(QWidget):
    def __init__(self):
        super().__init__()

        self.setObjectName("sidebar")

        # Start expanded
        self.setMinimumWidth(0)
        self.setMaximumWidth(0)


        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)

        title = QLabel("History")
        title.setStyleSheet("font-size: 14px; font-weight: 600;")
        layout.addWidget(title)

        layout.addStretch()

        # Animations
        self._min_anim = QPropertyAnimation(self, b"minimumWidth")
        self._max_anim = QPropertyAnimation(self, b"maximumWidth")

        for anim in (self._min_anim, self._max_anim):
            anim.setDuration(250)
            anim.setEasingCurve(QEasingCurve.InOutCubic)

    def toggle(self, expand: bool):
        for anim in (self._min_anim, self._max_anim):
            anim.stop()

        if expand:
            self._min_anim.setStartValue(0)
            self._min_anim.setEndValue(250)

            self._max_anim.setStartValue(0)
            self._max_anim.setEndValue(250)
        else:
            self._min_anim.setStartValue(self.width())
            self._min_anim.setEndValue(0)

            self._max_anim.setStartValue(self.width())
            self._max_anim.setEndValue(0)

        self._min_anim.start()
        self._max_anim.start()
