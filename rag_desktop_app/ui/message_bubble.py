from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel, QGraphicsDropShadowEffect
from PySide6.QtGui import QFont, QColor


class MessageBubble(QFrame):
    """
    Reusable message bubble widget for user and AI messages.
    Pure UI component – no business logic.
    """

    def __init__(self, message: str, is_user: bool = False, timestamp: str | None = None):
        super().__init__()
        self.is_user = is_user
        self._setup_ui(message, timestamp)
        self._add_shadow_effect()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------
    def _setup_ui(self, message: str, timestamp: str | None):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(6)

        # Main message text
        message_label = QLabel(message)
        message_label.setWordWrap(True)
        message_label.setFont(QFont("Segoe UI", 12))
        message_label.setStyleSheet("line-height: 1.4;")

        if self.is_user:
            self.setObjectName("userMessage")
            message_label.setStyleSheet(
                "color: white; line-height: 1.4;"
            )
        else:
            self.setObjectName("aiMessage")
            message_label.setStyleSheet(
                "color: #e8e8e8; line-height: 1.4;"
            )

        layout.addWidget(message_label)

        # Optional timestamp
        if timestamp:
            time_label = QLabel(timestamp)
            time_label.setFont(QFont("Segoe UI", 9))
            time_label.setStyleSheet(
                "color: rgba(200, 200, 200, 0.7);"
            )
            layout.addWidget(time_label)

    # ------------------------------------------------------------------
    # Visual polish
    # ------------------------------------------------------------------
    def _add_shadow_effect(self):
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(8)
        shadow.setColor(QColor(0, 0, 0, 25))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)
