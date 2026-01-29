from PySide6.QtWidgets import QMainWindow, QWidget, QHBoxLayout
from PySide6.QtCore import Qt, QPoint
from datetime import datetime

from rag_desktop_app.ui.sidebar import Sidebar
from rag_desktop_app.ui.chat_area import ChatArea
from rag_desktop_app.threads.rag_thread import RAGThread
from rag_desktop_app.services.persistence import ConversationStore
from rag_desktop_app.models.conversation import Conversation, Message


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowFlags(Qt.FramelessWindowHint)
        self.resize(1200, 800)

        self._drag_pos = QPoint()
        self._active_thread = None

        # ---------------- State ----------------
        self.store = ConversationStore()
        self.conversations = self.store.load_all()
        self.active_conversation: Conversation | None = None

        # ---------------- UI ----------------
        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self.sidebar = Sidebar()
        self.chat_area = ChatArea()

        layout.addWidget(self.sidebar)
        layout.addWidget(self.chat_area)
        self.setCentralWidget(central)

        # ---------------- Restore sidebar ----------------
        for c in self.conversations:
            self.sidebar.add_conversation(c)

        # ---------------- Signals ----------------
        self.chat_area.send_message.connect(self._on_message)
        self.sidebar.conversation_selected.connect(self._load_conversation)

    # --------------------------------------------------
    # Chat flow
    # --------------------------------------------------
    def _on_message(self, text: str):
        if self.active_conversation is None:
            self.active_conversation = Conversation(
                title=text[:30]
            )
            self.conversations.append(self.active_conversation)
            self.sidebar.add_conversation(self.active_conversation)

        self.chat_area.add_user_message(text)

        self.active_conversation.messages.append(
            Message(
                role="user",
                content=text,
                timestamp=datetime.now().strftime("%H:%M")
            )
        )

        self.chat_area.show_loading()

        self._active_thread = RAGThread(text)
        self._active_thread.finished.connect(self._on_rag_response)
        self._active_thread.error.connect(self._on_rag_error)
        self._active_thread.start()

    def _on_rag_response(self, data: dict):
        self.chat_area.hide_loading()

        answer = data.get("answer", "")
        self.chat_area.add_bot_message(answer)

        self.active_conversation.messages.append(
            Message(
                role="assistant",
                content=answer,
                timestamp=datetime.now().strftime("%H:%M")
            )
        )

        self.store.save_all(self.conversations)
        self._active_thread = None

    def _on_rag_error(self, error: str):
        self.chat_area.hide_loading()
        self.chat_area.add_system_message(error)
        self._active_thread = None

    # --------------------------------------------------
    # Load conversation
    # --------------------------------------------------
    def _load_conversation(self, conversation_id: str):
        for c in self.conversations:
            if c.id == conversation_id:
                self.active_conversation = c
                break

        self.chat_area.switch_to_chat()
        self.chat_area.messages_area.widget().deleteLater()

        # Rebuild messages
        from PySide6.QtWidgets import QWidget, QVBoxLayout
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addStretch()

        self.chat_area.messages_area.setWidget(container)
        self.chat_area._msg_layout = layout

        for m in self.active_conversation.messages:
            if m.role == "user":
                self.chat_area.add_user_message(m.content)
            else:
                self.chat_area.add_bot_message(m.content)
