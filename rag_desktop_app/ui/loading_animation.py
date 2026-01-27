from PySide6.QtWidgets import QWidget, QVBoxLayout, QProgressBar
from PySide6.QtCore import QTimer


class LoadingAnimation(QWidget):
    """
    Elegant indeterminate loading bar used while the AI is processing.
    Pure UI component – no business logic.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

        # Timer for breathing / pulsing effect
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._pulse_effect)

        self._opacity = 1.0
        self._direction = -1  # -1 = dim, 1 = brighten

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # indeterminate mode
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setObjectName("elegantProgress")

        layout.addWidget(self.progress_bar)
        self.setFixedSize(200, 20)

    # ------------------------------------------------------------------
    # Public controls
    # ------------------------------------------------------------------
    def start(self):
        self._timer.start(50)
        self.show()

    def stop(self):
        self._timer.stop()
        self.hide()

    # ------------------------------------------------------------------
    # Animation logic
    # ------------------------------------------------------------------
    def _pulse_effect(self):
        self._opacity += self._direction * 0.02

        if self._opacity <= 0.6:
            self._direction = 1
        elif self._opacity >= 1.0:
            self._direction = -1

        self.progress_bar.setStyleSheet(f"""
            QProgressBar#elegantProgress {{
                background-color: rgba(58, 58, 58, 0.8);
                border-radius: 3px;
                border: none;
            }}
            QProgressBar#elegantProgress::chunk {{
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 rgba(221, 160, 255, {self._opacity}),
                    stop: 0.5 rgba(187, 134, 252, {self._opacity}),
                    stop: 1 rgba(165, 112, 247, {self._opacity})
                );
                border-radius: 3px;
            }}
        """)
