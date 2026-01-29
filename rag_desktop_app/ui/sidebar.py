from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QPropertyAnimation
import qtawesome as qta

from rag_desktop_app.models.conversation import Conversation


class Sidebar(QWidget):
    conversation_selected = Signal(str)
    new_chat_requested = Signal()
    search_requested = Signal()

    RAIL_WIDTH = 56
    EXPANDED_WIDTH = 260

    def __init__(self):
        super().__init__()

        self.setObjectName("sidebar")
        self._expanded = False
        self._chat_buttons = {}

        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setMinimumWidth(self.RAIL_WIDTH)
        self.setMaximumWidth(self.RAIL_WIDTH)

        self._setup_ui()

    # ======================================================
    # UI SETUP
    # ======================================================
    def _setup_ui(self):
        self.root_layout = QVBoxLayout(self)
        self.root_layout.setContentsMargins(0, 0, 0, 0)
        self.root_layout.setSpacing(0)

        # ==================================================
        # ACTIONS (ICON + TEXT)
        # ==================================================
        self.actions_container = QWidget()
        actions_layout = QVBoxLayout(self.actions_container)
        actions_layout.setContentsMargins(8, 8, 8, 8)
        actions_layout.setSpacing(10)

        self.new_chat_btn, self.new_chat_text = self._create_action(
            "fa5s.plus", "New Chat", self.new_chat_requested.emit
        )
        self.search_btn, self.search_text = self._create_action(
            "fa5s.search", "Search", self.search_requested.emit
        )

        actions_layout.addWidget(self.new_chat_btn)
        actions_layout.addWidget(self.search_btn)

        self.root_layout.addWidget(self.actions_container)

        # ==================================================
        # CHAT LIST
        # ==================================================
        self.expand_container = QWidget()
        self.expand_container.hide()

        expand_layout = QVBoxLayout(self.expand_container)
        expand_layout.setContentsMargins(12, 6, 12, 12)
        expand_layout.setSpacing(10)

        self.section_label = QLabel("Chats")
        self.section_label.setObjectName("sidebarSection")
        expand_layout.addWidget(self.section_label)

        self.chat_list_container = QWidget()
        self.chat_list_layout = QVBoxLayout(self.chat_list_container)
        self.chat_list_layout.setSpacing(8)
        self.chat_list_layout.setContentsMargins(0, 0, 0, 0)
        self.chat_list_layout.addStretch()

        expand_layout.addWidget(self.chat_list_container)

        self.root_layout.addWidget(self.expand_container)
        self.root_layout.addStretch()

    # ======================================================
    # ACTION BUTTON BUILDER
    # ======================================================
    def _create_action(self, icon_name, text, callback):
        btn = QPushButton()
        btn.setObjectName("sidebarAction")
        btn.setCursor(Qt.PointingHandCursor)

        layout = QHBoxLayout(btn)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(12)

        icon = QLabel()
        icon.setPixmap(qta.icon(icon_name).pixmap(16, 16))

        label = QLabel(text)
        label.setObjectName("sidebarActionText")
        label.hide()  # hidden when collapsed

        layout.addWidget(icon)
        layout.addWidget(label)

        btn.clicked.connect(callback)

        return btn, label

    # ======================================================
    # COLLAPSE / EXPAND
    # ======================================================
    def toggle(self, expand: bool):
        self._expanded = expand

        self.expand_container.setVisible(expand)
        self.new_chat_text.setVisible(expand)
        self.search_text.setVisible(expand)

        start = self.minimumWidth()
        end = self.EXPANDED_WIDTH if expand else self.RAIL_WIDTH

        self.anim = QPropertyAnimation(self, b"minimumWidth")
        self.anim.setDuration(220)
        self.anim.setStartValue(start)
        self.anim.setEndValue(end)
        self.anim.start()

    # ======================================================
    # CONVERSATIONS
    # ======================================================
    def add_conversation(self, conversation: Conversation):
        if conversation.id in self._chat_buttons:
            return

        btn = QPushButton(conversation.title)
        btn.setObjectName("chatItem")
        btn.setCheckable(True)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setMinimumHeight(38)

        btn.clicked.connect(
            lambda: self.conversation_selected.emit(conversation.id)
        )

        self.chat_list_layout.insertWidget(
            self.chat_list_layout.count() - 1,
            btn
        )

        self._chat_buttons[conversation.id] = btn

    def set_active(self, conversation_id: str):
        for cid, btn in self._chat_buttons.items():
            btn.setChecked(cid == conversation_id)
