from datetime import datetime
from typing import Optional, List

from PySide6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QSplitter
from PySide6.QtCore import Qt

from rag_desktop_app.ui.sidebar import Sidebar
from rag_desktop_app.ui.chat_area import ChatArea
from rag_desktop_app.threads.rag_thread import RAGThread
from rag_desktop_app.models.conversation import Conversation
from rag_desktop_app.services.persistence import ConversationStorage


class MainWindow(QMainWindow):
    """
    Main application window.
    Orchestrates UI components, state, threading, and persistence.
    """

    def __init__(self):
        super().__init__()

        self.setWindowTitle("RAG Assistant")
        self.setMinimumSize(1000, 600)

        # --------------------------------------------------------------
        # State
        # --------------------------------------------------------------
        self.conversations: List[Conversation] = []
        self.current_conversation: Optional[Conversation] = None
        self.storage = ConversationStorage()
        self.rag_thread: Optional[RAGThread] = None

        # --------------------------------------------------------------
        # UI setup
        # --------------------------------------------------------------
        self._setup_ui()

        # --------------------------------------------------------------
        # Load persisted data
        # --------------------------------------------------------------
        self.conversations = self.storage.load()
        if self.conversations:
            self.current_conversation = self.conversations[0]

        self._refresh_ui()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)

        # Sidebar
        self.sidebar = Sidebar()
        self.sidebar.new_conversation_requested.connect(self._create_conversation)
        self.sidebar.conversation_selected.connect(self._select_conversation)

        # Chat area
        self.chat_area = ChatArea()
        self.chat_area.send_requested.connect(self._send_message)

        splitter.addWidget(self.sidebar)
        splitter.addWidget(self.chat_area)
        splitter.setSizes([300, 900])
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)

        layout.addWidget(splitter)

    # ------------------------------------------------------------------
    # Conversation handling
    # ------------------------------------------------------------------
    def _create_conversation(self):
        conv = Conversation()
        self.conversations.insert(0, conv)
        self.current_conversation = conv
        self._refresh_ui()
        self._persist()

    def _select_conversation(self, conversation: Conversation):
        self.current_conversation = conversation
        self._refresh_ui()

    def _refresh_ui(self):
        self.sidebar.update_conversations(
            self.conversations,
            self.current_conversation
        )

        self.chat_area.clear_messages()

        if not self.current_conversation:
            self.chat_area.set_title("Select or create a conversation")
            return

        self.chat_area.set_title(self.current_conversation.title)

        for msg in self.current_conversation.messages:
            self.chat_area.add_message(
                msg["content"],
                msg["is_user"],
                msg.get("timestamp")
            )

    # ------------------------------------------------------------------
    # Message flow
    # ------------------------------------------------------------------
    def _send_message(self, text: str):
        if not self.current_conversation:
            self._create_conversation()

        timestamp = datetime.now().strftime("%H:%M")

        # Add user message
        self.current_conversation.add_message(text, True, timestamp)
        self.chat_area.add_message(text, True, timestamp)

        self.chat_area.show_loading()

        # Start background thread
        self.rag_thread = RAGThread(text)
        self.rag_thread.finished.connect(self._on_response)
        self.rag_thread.error.connect(self._on_error)
        self.rag_thread.start()

        self._persist()

    def _on_response(self, response: dict):
        self.chat_area.hide_loading()

        timestamp = datetime.now().strftime("%H:%M")
        answer = response.get("answer", "")

        # Add AI message
        self.current_conversation.add_message(answer, False, timestamp)
        self.chat_area.add_message(answer, False, timestamp)

        # Update evaluation + pipeline + context
        self.chat_area.update_evaluation(response.get("evaluation", {}))
        self.chat_area.update_pipeline(response.get("pipeline", {}))
        self.chat_area.show_context(response.get("context_used", []))

        self._persist()
        self._refresh_ui()

    def _on_error(self, error: str):
        self.chat_area.hide_loading()
        self.chat_area.add_message(f"Error: {error}", False)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _persist(self):
        self.storage.save(self.conversations)

    def closeEvent(self, event):
        self._persist()
        event.accept()
