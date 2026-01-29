from PySide6.QtWidgets import QMainWindow, QWidget, QHBoxLayout
from PySide6.QtCore import Qt, QPoint

from rag_desktop_app.ui.sidebar import Sidebar
from rag_desktop_app.ui.chat_area import ChatArea


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowFlags(Qt.FramelessWindowHint)
        self.resize(1200, 800)

        self._drag_pos = QPoint()
        self._sidebar_expanded = False
        self._chat_started = False

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self.sidebar = Sidebar()
        self.chat_area = ChatArea()

        layout.addWidget(self.sidebar)
        layout.addWidget(self.chat_area)
        self.setCentralWidget(central)

        # Drag
        h = self.chat_area.chat_header
        h.mousePressEvent = self._mouse_press
        h.mouseMoveEvent = self._mouse_move

        # Sidebar
        self.chat_area.menu_button.clicked.connect(self.toggle_sidebar)

        # Chat state
        self.chat_area.send_message.connect(self._on_message)

    def toggle_sidebar(self):
        self._sidebar_expanded = not self._sidebar_expanded
        self.sidebar.toggle(self._sidebar_expanded)

    def _on_message(self, text: str):
        if not self._chat_started:
            self._chat_started = True
            self.chat_area.switch_to_chat()

    def _mouse_press(self, e):
        if e.button() == Qt.LeftButton:
            self._drag_pos = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def _mouse_move(self, e):
        if e.buttons() == Qt.LeftButton:
            self.move(e.globalPosition().toPoint() - self._drag_pos)
