from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel,
    QPushButton, QListWidget, QListWidgetItem
)
from PySide6.QtGui import QFont
from PySide6.QtCore import Signal

from models.conversation import Conversation, ConversationItem


class Sidebar(QWidget):
    """
    Left sidebar containing:
    - App title
    - New conversation button
    - Conversation list
    Emits signals instead of handling logic directly.
    """

    new_conversation_requested = Signal()
    conversation_selected = Signal(Conversation)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------
    def _setup_ui(self):
        self.setObjectName("sidebar")
        self.setFixedWidth(300)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # --------------------------------------------------------------
        # Header
        # --------------------------------------------------------------
        header = QWidget()
        header_layout = QVBoxLayout(header)
        header_layout.setSpacing(4)

        title = QLabel("RAG Assistant")
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        title.setStyleSheet("color: #ffffff;")

        subtitle = QLabel("AI Document Analysis")
        subtitle.setFont(QFont("Segoe UI", 11))
        subtitle.setStyleSheet("color: #b0b0b0;")

        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)

        layout.addWidget(header)

        # --------------------------------------------------------------
        # New conversation button
        # --------------------------------------------------------------
        self.new_chat_btn = QPushButton("✨ New Conversation")
        self.new_chat_btn.setFixedHeight(42)
        self.new_chat_btn.setObjectName("newChatButton")
        self.new_chat_btn.clicked.connect(self.new_conversation_requested.emit)

        layout.addWidget(self.new_chat_btn)

        # --------------------------------------------------------------
        # Conversation list
        # --------------------------------------------------------------
        self.conversation_list = QListWidget()
        self.conversation_list.setObjectName("conversationsList")
        self.conversation_list.itemClicked.connect(self._on_item_clicked)

        layout.addWidget(self.conversation_list)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update_conversations(self, conversations: list[Conversation], current: Conversation | None):
        self.conversation_list.clear()

        for conv in conversations:
            item = QListWidgetItem()
            widget = ConversationItem(conv)
            item.setSizeHint(widget.sizeHint())

            self.conversation_list.addItem(item)
            self.conversation_list.setItemWidget(item, widget)

        if current and current in conversations:
            index = conversations.index(current)
            self.conversation_list.setCurrentRow(index)

    # ------------------------------------------------------------------
    # Internal handlers
    # ------------------------------------------------------------------
    def _on_item_clicked(self, item: QListWidgetItem):
        row = self.conversation_list.row(item)
        widget = self.conversation_list.itemWidget(item)

        if widget:
            self.conversation_selected.emit(widget.conversation)
