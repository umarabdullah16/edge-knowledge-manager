from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton
from PySide6.QtCore import Signal

from rag_desktop_app.models.conversation import Conversation


class Sidebar(QWidget):
    conversation_selected = Signal(str)

    def __init__(self):
        super().__init__()
        self._setup_ui()

    def _setup_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(6)
        self.layout.addStretch()

        self._buttons = {}

    def add_conversation(self, conversation: Conversation):
        if conversation.id in self._buttons:
            return

        btn = QPushButton(conversation.title)
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(
            lambda: self.conversation_selected.emit(conversation.id)
        )

        self.layout.insertWidget(0, btn)
        self._buttons[conversation.id] = btn

    def clear(self):
        for btn in self._buttons.values():
            btn.deleteLater()
        self._buttons.clear()
