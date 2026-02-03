from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QFileDialog,
    QMessageBox
)
from PySide6.QtCore import Qt, QPoint
from datetime import datetime
import uuid

from rag_desktop_app.ui.sidebar import Sidebar
from rag_desktop_app.ui.chat_area import ChatArea
from rag_desktop_app.threads.rag_thread import RAGThread
from rag_desktop_app.services.rag_backend import RAGBackend
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
        self._active_request_id: str | None = None

        # ==================================================
        # STATE
        # ==================================================
        self.store = ConversationStore()
        self.conversations = self.store.load_all()
        self.active_conversation: Conversation | None = None

        # Backend API client
        self.backend = RAGBackend()

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

        for c in self.conversations:
            self.sidebar.add_conversation(c)

        # ==================================================
        # SIGNALS
        # ==================================================
        self.chat_area.menu_button.clicked.connect(self.toggle_sidebar)
        self.sidebar.conversation_selected.connect(self._load_conversation)
        self.sidebar.new_chat_requested.connect(self._on_new_chat)
        self.sidebar.conversation_delete_requested.connect(
            self._delete_conversation
        )

        self.chat_area.upload_requested.connect(self._on_upload_requested)
        self.chat_area.send_message.connect(self._on_message)

        self.chat_area.minimize_requested.connect(self.showMinimized)
        self.chat_area.close_requested.connect(self.close)
        self.chat_area.maximize_requested.connect(self._toggle_maximize)

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
        self._active_request_id = None
        self.active_conversation = None
        self.sidebar.set_active("")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(14)
        layout.addStretch()

        self.chat_area.messages_area.setWidget(container)
        self.chat_area._msg_layout = layout
        self.chat_area.reset_to_start()

    # ======================================================
    # DELETE CONVERSATION
    # ======================================================
    def _delete_conversation(self, conversation_id: str):
        reply = QMessageBox.question(
            self,
            "Delete Chat",
            "Are you sure you want to delete this conversation?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        self.conversations = [
            c for c in self.conversations
            if c.id != conversation_id
        ]

        self.sidebar.remove_conversation(conversation_id)
        self.store.save_all(self.conversations)

        if (
            self.active_conversation
            and self.active_conversation.id == conversation_id
        ):
            self.active_conversation = None
            self._active_request_id = None
            self._on_new_chat()

    # ======================================================
    # UPLOAD / INGEST  ✅ UPDATED
    # ======================================================
    def _on_upload_requested(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select PDF files",
            "",
            "PDF Files (*.pdf)"
        )

        if not files:
            return

        self._run_ingest(files)

    def _run_ingest(self, files: list[str]):
        try:
            self.chat_area.add_system_message("📥 Indexing documents… Please wait.")

            for file_path in files:
                self.backend.ingest_document(file_path)

            QMessageBox.information(
                self,
                "Ingestion Complete",
                f"{len(files)} PDF file(s) successfully ingested."
            )

        except RuntimeError as e:
            QMessageBox.critical(
                self,
                "Ingestion Failed",
                str(e)
            )

    # ======================================================
    # CHAT FLOW
    # ======================================================
    def _on_message(self, text: str):
        if self.active_conversation is None:
            self.active_conversation = Conversation(title=text[:30])
            self.conversations.append(self.active_conversation)
            self.sidebar.add_conversation(self.active_conversation)
            self.store.save_all(self.conversations)

        self.sidebar.set_active(self.active_conversation.id)

        self.chat_area.add_user_message(text)
        self.chat_area.on_query_started()

        self.active_conversation.messages.append(
            Message(
                role="user",
                content=text,
                timestamp=datetime.now().strftime("%H:%M")
            )
        )
        self.store.save_all(self.conversations)

        request_id = uuid.uuid4().hex
        self._active_request_id = request_id

        thread = RAGThread(text)
        self._active_thread = thread

        def handle_response(data: dict):
            if self._active_request_id != request_id:
                return

            answer = data.get("answer", "")
            self.chat_area.add_bot_message(answer)
            self.chat_area.on_query_finished()

            self.active_conversation.messages.append(
                Message(
                    role="assistant",
                    content=answer,
                    timestamp=datetime.now().strftime("%H:%M")
                )
            )
            self.store.save_all(self.conversations)
            self._active_thread = None

        def handle_error(_):
            if self._active_request_id != request_id:
                return

            self.chat_area.add_system_message(
                "⚠ Unable to connect to the AI service.\n"
                "Please make sure the backend server is running."
            )
            self.chat_area.on_query_finished()
            self._active_thread = None

        thread.finished.connect(handle_response)
        thread.error.connect(handle_error)
        thread.start()

    # ======================================================
    # LOAD CONVERSATION
    # ======================================================
    def _load_conversation(self, conversation_id: str):
        for c in self.conversations:
            if c.id == conversation_id:
                self.active_conversation = c
                break
        else:
            return

        self._active_request_id = None
        self.sidebar.set_active(conversation_id)
        self.chat_area.switch_to_chat()

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(14)
        layout.addStretch()

        self.chat_area.messages_area.setWidget(container)
        self.chat_area._msg_layout = layout

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
