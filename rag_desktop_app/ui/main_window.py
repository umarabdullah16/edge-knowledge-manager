from PySide6.QtWidgets import QMainWindow, QWidget, QHBoxLayout
from PySide6.QtCore import Qt, QPoint

from rag_desktop_app.ui.sidebar import Sidebar
from rag_desktop_app.ui.chat_area import ChatArea
from rag_desktop_app.threads.rag_thread import RAGThread


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowFlags(Qt.FramelessWindowHint)
        self.resize(1200, 800)

        self._drag_pos = QPoint()
        self._sidebar_expanded = False
        self._chat_started = False
        self._active_thread = None  # important: prevent GC

        # ---------------- UI setup ----------------
        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self.sidebar = Sidebar()
        self.chat_area = ChatArea()

        layout.addWidget(self.sidebar)
        layout.addWidget(self.chat_area)
        self.setCentralWidget(central)

        # ---------------- Drag handling ----------------
        h = self.chat_area.chat_header
        h.mousePressEvent = self._mouse_press
        h.mouseMoveEvent = self._mouse_move

        # ---------------- Signals ----------------
        self.chat_area.menu_button.clicked.connect(self.toggle_sidebar)
        self.chat_area.send_message.connect(self._on_message)

    # ------------------------------------------------
    # Sidebar
    # ------------------------------------------------
    def toggle_sidebar(self):
        self._sidebar_expanded = not self._sidebar_expanded
        self.sidebar.toggle(self._sidebar_expanded)

    # ------------------------------------------------
    # Chat flow (CORE)
    # ------------------------------------------------
    def _on_message(self, text: str):
        if not text.strip():
            return

        if not self._chat_started:
            self._chat_started = True
            self.chat_area.switch_to_chat()

        # 1️⃣ Show user message
        self.chat_area.add_user_message(text)

        # 2️⃣ Show loading indicator
        self.chat_area.show_loading()

        # 3️⃣ Start backend thread
        self._active_thread = RAGThread(text)
        self._active_thread.finished.connect(self._on_rag_response)
        self._active_thread.error.connect(self._on_rag_error)
        self._active_thread.start()

    def _on_rag_response(self, data: dict):
        """
        data format (UI-safe):
        {
            "answer": str,
            "sources": list,
            "evaluation": dict | None,
            "latency_ms": int | None,
            "raw": dict
        }
        """
        self.chat_area.hide_loading()

        self.chat_area.add_bot_message(
            text=data.get("answer", ""),
            sources=data.get("sources"),
            latency=data.get("latency_ms"),
        )

        self._active_thread = None

    def _on_rag_error(self, error_msg: str):
        self.chat_area.hide_loading()

        self.chat_area.add_system_message(
            f"⚠️ {error_msg}"
        )

        self._active_thread = None

    # ------------------------------------------------
    # Window dragging
    # ------------------------------------------------
    def _mouse_press(self, e):
        if e.button() == Qt.LeftButton:
            self._drag_pos = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def _mouse_move(self, e):
        if e.buttons() == Qt.LeftButton:
            self.move(e.globalPosition().toPoint() - self._drag_pos)
