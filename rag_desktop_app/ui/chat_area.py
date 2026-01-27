from datetime import datetime
from typing import Optional, Dict, List

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton,
    QScrollArea, QMessageBox
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont

from ui.message_bubble import MessageBubble
from ui.loading_animation import LoadingAnimation


class ChatArea(QWidget):
    """
    Main chat + evaluation area.
    Pure UI component:
    - Displays messages
    - Shows loading state
    - Exposes update hooks for evaluation, context, pipeline
    """

    send_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # --------------------------------------------------------------
        # Header
        # --------------------------------------------------------------
        header = QWidget()
        header.setFixedHeight(60)
        header.setObjectName("chatHeader")

        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(24, 0, 24, 0)

        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet("color: #4CAF50; font-size: 12px;")

        self.chat_title = QLabel("Select or create a conversation")
        self.chat_title.setFont(QFont("Segoe UI", 15, QFont.Medium))
        self.chat_title.setStyleSheet(
            "color: #ffffff; border-bottom: 1px solid rgba(187, 134, 252, 0.3);"
        )

        header_layout.addWidget(self.status_dot)
        header_layout.addWidget(self.chat_title)
        header_layout.addStretch()

        layout.addWidget(header)

        # --------------------------------------------------------------
        # Messages area
        # --------------------------------------------------------------
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setObjectName("messagesArea")

        self.messages_widget = QWidget()
        self.messages_layout = QVBoxLayout(self.messages_widget)
        self.messages_layout.setContentsMargins(24, 24, 24, 24)
        self.messages_layout.setSpacing(16)
        self.messages_layout.addStretch()

        self.scroll_area.setWidget(self.messages_widget)
        layout.addWidget(self.scroll_area)

        # --------------------------------------------------------------
        # Loading animation
        # --------------------------------------------------------------
        self.loading_container = QWidget()
        self.loading_container.setFixedHeight(60)
        self.loading_container.setObjectName("loadingContainer")
        self.loading_container.hide()

        loading_layout = QVBoxLayout(self.loading_container)
        loading_layout.setContentsMargins(24, 15, 24, 15)

        loading_label = QLabel("AI is processing your request...")
        loading_label.setFont(QFont("Segoe UI", 12, QFont.Medium))
        loading_label.setStyleSheet("color: #dda0ff;")
        loading_label.setAlignment(Qt.AlignCenter)

        self.loading_animation = LoadingAnimation()

        loading_layout.addWidget(loading_label, 0, Qt.AlignCenter)
        loading_layout.addWidget(self.loading_animation, 0, Qt.AlignCenter)

        layout.addWidget(self.loading_container)

        # --------------------------------------------------------------
        # Evaluation / pipeline info
        # --------------------------------------------------------------
        self.info_label = QLabel("Evaluation: Not computed")
        self.info_label.setStyleSheet(
            "color: #dda0ff; font-size: 12px; padding: 6px 24px;"
        )
        layout.addWidget(self.info_label)

        # --------------------------------------------------------------
        # Input area
        # --------------------------------------------------------------
        input_area = QWidget()
        input_area.setFixedHeight(90)
        input_area.setObjectName("inputArea")

        input_layout = QHBoxLayout(input_area)
        input_layout.setContentsMargins(24, 16, 24, 16)
        input_layout.setSpacing(16)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Ask about your documents...")
        self.input_field.setFont(QFont("Segoe UI", 13))
        self.input_field.setObjectName("inputField")
        self.input_field.returnPressed.connect(self._on_send_clicked)

        self.send_button = QPushButton("Send")
        self.send_button.setFixedSize(90, 42)
        self.send_button.setObjectName("sendButton")
        self.send_button.clicked.connect(self._on_send_clicked)

        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_button)

        layout.addWidget(input_area)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_title(self, title: str):
        self.chat_title.setText(title)

    def add_message(self, text: str, is_user: bool, timestamp: Optional[str] = None):
        bubble = MessageBubble(text, is_user, timestamp)

        container = QWidget()
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        if is_user:
            container_layout.addStretch()
            container_layout.addWidget(bubble)
        else:
            container_layout.addWidget(bubble)
            container_layout.addStretch()

        self.messages_layout.insertWidget(
            self.messages_layout.count() - 1,
            container
        )

        QTimer.singleShot(100, self._scroll_to_bottom)

    def clear_messages(self):
        for i in reversed(range(self.messages_layout.count() - 1)):
            item = self.messages_layout.itemAt(i)
            if item.widget():
                item.widget().deleteLater()

    def show_loading(self):
        self.loading_container.show()
        self.loading_animation.start()
        self._set_input_enabled(False)

    def hide_loading(self):
        self.loading_container.hide()
        self.loading_animation.stop()
        self._set_input_enabled(True)

    def update_evaluation(self, evaluation: Dict):
        if not evaluation or all(v is None for v in evaluation.values()):
            self.info_label.setText("Evaluation: Not computed")
            return

        text = "Evaluation | "
        for key, value in evaluation.items():
            if value is not None:
                text += f"{key.replace('_',' ').title()}: {round(value * 100, 1)}%  "

        self.info_label.setText(text)

    def update_pipeline(self, pipeline: Dict):
        if not pipeline:
            return

        tooltip = (
            f"Retrieval: {pipeline.get('retrieval_ms', '—')} ms | "
            f"Generation: {pipeline.get('generation_ms', '—')} ms | "
            f"Total: {pipeline.get('total_ms', '—')} ms"
        )
        self.status_dot.setToolTip(tooltip)

    def show_context(self, context: List[Dict]):
        if not context:
            return

        text = "Context Used:\n\n"
        for idx, chunk in enumerate(context, start=1):
            text += f"[{idx}] {chunk.get('source', 'unknown')}\n"
            text += chunk.get("content", "")[:300] + "\n\n"

        QMessageBox.information(self, "Context Used", text)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _on_send_clicked(self):
        message = self.input_field.text().strip()
        if not message:
            return

        self.input_field.clear()
        self.send_requested.emit(message)

    def _scroll_to_bottom(self):
        bar = self.scroll_area.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _set_input_enabled(self, enabled: bool):
        self.input_field.setEnabled(enabled)
        self.send_button.setEnabled(enabled)
