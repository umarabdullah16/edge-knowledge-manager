from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSizePolicy,
    QScrollArea,
    QToolButton
)
from PySide6.QtCore import Qt, Signal, QPropertyAnimation, QSize
import qtawesome as qta

from rag_desktop_app.models.conversation import Conversation


class Sidebar(QWidget):
    conversation_selected = Signal(str)
    conversation_delete_requested = Signal(str)
    new_chat_requested = Signal()
    search_requested = Signal()
    documents_requested = Signal()

    RAIL_WIDTH = 56
    EXPANDED_WIDTH = 260

    def __init__(self):
        super().__init__()

        self.setObjectName("sidebar")
        self._expanded = False
        self._chat_widgets = {}

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
        # ACTIONS (ICON RAIL)
        # ==================================================
        self.actions_container = QWidget()
        self.actions_layout = QVBoxLayout(self.actions_container)
        self.actions_layout.setContentsMargins(6, 6, 6, 6)
        self.actions_layout.setSpacing(6)

        self.new_chat_btn, self.new_chat_text = self._create_action(
            "fa5s.plus", "New Chat", self.new_chat_requested.emit
        )
        self.search_btn, self.search_text = self._create_action(
            "fa5s.search", "Search", self.search_requested.emit
        )

        # 📄 DOCUMENTS ACTION (NEW)
        self.docs_btn, self.docs_text = self._create_action(
            "fa5s.file-alt", "Documents", self.documents_requested.emit
        )

        # Stats label (only visible when expanded)
        self.docs_stats_label = QLabel("Documents indexed: 0\nTotal chunks: 0")
        self.docs_stats_label.setObjectName("sidebarDocStats")
        self.docs_stats_label.setAlignment(Qt.AlignLeft)
        self.docs_stats_label.hide()

        self.actions_layout.addWidget(self.new_chat_btn)
        self.actions_layout.addWidget(self.search_btn)
        self.actions_layout.addWidget(self.docs_btn)
        self.actions_layout.addWidget(self.docs_stats_label)
        self.actions_layout.addStretch()

        self.root_layout.addWidget(self.actions_container)

        # ==================================================
        # EXPANDABLE CHAT LIST (UNCHANGED)
        # ==================================================
        self.expand_container = QWidget()
        self.expand_container.hide()

        expand_layout = QVBoxLayout(self.expand_container)
        expand_layout.setContentsMargins(12, 6, 12, 12)
        expand_layout.setSpacing(10)

        self.section_label = QLabel("Chats")
        self.section_label.setObjectName("sidebarSection")
        expand_layout.addWidget(self.section_label)

        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.chat_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.chat_scroll.setFrameShape(QScrollArea.NoFrame)

        self.chat_list_container = QWidget()
        self.chat_list_layout = QVBoxLayout(self.chat_list_container)
        self.chat_list_layout.setSpacing(6)
        self.chat_list_layout.setContentsMargins(0, 0, 0, 0)
        self.chat_list_layout.addStretch()

        self.chat_scroll.setWidget(self.chat_list_container)
        expand_layout.addWidget(self.chat_scroll, 1)

        self.root_layout.addWidget(self.expand_container, 1)

    # ======================================================
    # ACTION BUTTON BUILDER
    # ======================================================
    def _create_action(self, icon_name, text, callback):
        btn = QPushButton()
        btn.setObjectName("sidebarAction")
        btn.setCursor(Qt.PointingHandCursor)
        btn.setMinimumHeight(36)

        layout = QHBoxLayout(btn)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(10)

        icon = QLabel()
        icon.setPixmap(qta.icon(icon_name).pixmap(16, 16))

        label = QLabel(text)
        label.setObjectName("sidebarActionText")
        label.hide()

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
        self.docs_text.setVisible(expand)
        self.docs_stats_label.setVisible(expand)

        if expand:
            self.actions_layout.setContentsMargins(8, 8, 8, 8)
            self.actions_layout.setSpacing(10)
        else:
            self.actions_layout.setContentsMargins(6, 6, 6, 6)
            self.actions_layout.setSpacing(6)

        start = self.minimumWidth()
        end = self.EXPANDED_WIDTH if expand else self.RAIL_WIDTH

        self.anim = QPropertyAnimation(self, b"minimumWidth")
        self.anim.setDuration(220)
        self.anim.setStartValue(start)
        self.anim.setEndValue(end)
        self.anim.start()

    # ======================================================
    # DOCUMENT STATS (NEW)
    # ======================================================
    def update_document_stats(self, total_docs: int, total_chunks: int):
        self.docs_stats_label.setText(
            f"Documents indexed: {total_docs}\n"
            f"Total chunks: {total_chunks}"
        )

    # ======================================================
    # CONVERSATIONS (UNCHANGED)
    # ======================================================
    def add_conversation(self, conversation: Conversation):
        if conversation.id in self._chat_widgets:
            return

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(6, 2, 6, 2)
        row_layout.setSpacing(6)

        chat_btn = QPushButton(conversation.title)
        chat_btn.setObjectName("chatItem")
        chat_btn.setCheckable(True)
        chat_btn.setCursor(Qt.PointingHandCursor)
        chat_btn.setMinimumHeight(34)
        chat_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        chat_btn.setMaximumWidth(180)
        chat_btn.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding-left: 10px;
                padding-right: 6px;
            }
        """)

        chat_btn.clicked.connect(
            lambda: self.conversation_selected.emit(conversation.id)
        )

        delete_btn = QToolButton()
        delete_btn.setFixedSize(22, 22)
        delete_btn.setIconSize(QSize(14, 14))
        delete_btn.setIcon(qta.icon("fa5s.trash", color="#ff6b6b"))
        delete_btn.setCursor(Qt.PointingHandCursor)
        delete_btn.setAutoRaise(True)
        delete_btn.setToolTip("Delete chat")
        delete_btn.clicked.connect(
            lambda: self.conversation_delete_requested.emit(conversation.id)
        )

        row_layout.addWidget(chat_btn)
        row_layout.addWidget(delete_btn)

        self.chat_list_layout.insertWidget(
            self.chat_list_layout.count() - 1,
            row
        )

        self._chat_widgets[conversation.id] = (row, chat_btn)

    def remove_conversation(self, conversation_id: str):
        if conversation_id not in self._chat_widgets:
            return

        row, _ = self._chat_widgets.pop(conversation_id)
        row.setParent(None)
        row.deleteLater()

    def set_active(self, conversation_id: str):
        for cid, (_, btn) in self._chat_widgets.items():
            btn.setChecked(cid == conversation_id)
