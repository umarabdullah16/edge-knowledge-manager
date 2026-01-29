from PySide6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout
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

        # ==================================================
        # WINDOW SETUP
        # ==================================================
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.resize(1200, 800)

        self._drag_pos = QPoint()
        self._sidebar_expanded = False
        self._active_thread = None

        # ==================================================
        # STATE
        # ==================================================
        self.store = ConversationStore()
        self.conversations = self.store.load_all()
        self.active_conversation: Conversation | None = None

        # ==================================================
        # UI
        # ==================================================
        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.sidebar = Sidebar()
        self.chat_area = ChatArea()

        layout.addWidget(self.sidebar)
        layout.addWidget(self.chat_area)
        self.setCentralWidget(central)

        # Restore conversations into sidebar
        for c in self.conversations:
            self.sidebar.add_conversation(c)

        # ==================================================
        # SIGNALS
        # ==================================================

        # Sidebar
        self.chat_area.menu_button.clicked.connect(self.toggle_sidebar)
        self.sidebar.conversation_selected.connect(self._load_conversation)
        self.sidebar.new_chat_requested.connect(self._on_new_chat)

        # Chat
        self.chat_area.send_message.connect(self._on_message)

        # Window controls
        self.chat_area.minimize_requested.connect(self.showMinimized)
        self.chat_area.close_requested.connect(self.close)
        self.chat_area.maximize_requested.connect(self._toggle_maximize)

        # Drag window from header
        header = self.chat_area.chat_header
        header.mousePressEvent = self._mouse_press
        header.mouseMoveEvent = self._mouse_move

    # ======================================================
    # SIDEBAR
    # ======================================================
    def toggle_sidebar(self):
        self._sidebar_expanded = not self._sidebar_expanded
        self.sidebar.toggle(self._sidebar_expanded)

    def _on_new_chat(self):
        """
        User explicitly starts a new chat.
        Conversation is created lazily on first message.
        """
        self.active_conversation = None
        self.sidebar.set_active("")
        self.chat_area.reset_to_start()

    # ======================================================
    # CHAT FLOW
    # ======================================================
    def _on_message(self, text: str):
        """
        Handle user message.
        """
        # Lazily create conversation
        if self.active_conversation is None:
            self.active_conversation = Conversation(title=text[:30])
            self.conversations.append(self.active_conversation)
            self.sidebar.add_conversation(self.active_conversation)
            self.store.save_all(self.conversations)

        self.sidebar.set_active(self.active_conversation.id)

        # UI updates
        self.chat_area.add_user_message(text)
        self.chat_area.show_loading()

        # Persist user message
        self.active_conversation.messages.append(
            Message(
                role="user",
                content=text,
                timestamp=datetime.now().strftime("%H:%M")
            )
        )
        self.store.save_all(self.conversations)

        # Start RAG backend thread
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
        self.store.save_all(self.conversations)
        self._active_thread = None

    # ======================================================
    # LOAD CONVERSATION
    # ======================================================
    def _load_conversation(self, conversation_id: str):
        """
        Load an existing conversation into the chat area.
        """
        for c in self.conversations:
            if c.id == conversation_id:
                self.active_conversation = c
                break
        else:
            return

        self.sidebar.set_active(conversation_id)

        # Switch to chat view
        self.chat_area.switch_to_chat()

        # Clear existing messages
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(14)
        layout.addStretch()

        self.chat_area.messages_area.setWidget(container)
        self.chat_area._msg_layout = layout

        # Render conversation messages
        for m in self.active_conversation.messages:
            if m.role == "user":
                self.chat_area.add_user_message(m.content)
            else:
                self.chat_area.add_bot_message(m.content)

    # ======================================================
    # WINDOW CONTROLS
    # ======================================================
    def _toggle_maximize(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    # ======================================================
    # WINDOW DRAG
    # ======================================================
    def _mouse_press(self, e):
        if e.button() == Qt.LeftButton:
            self._drag_pos = (
                e.globalPosition().toPoint()
                - self.frameGeometry().topLeft()
            )

    def _mouse_move(self, e):
        if e.buttons() == Qt.LeftButton:
            self.move(
                e.globalPosition().toPoint()
                - self._drag_pos
            )
