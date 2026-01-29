from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QSizePolicy
from PySide6.QtCore import Signal, Qt, QPropertyAnimation

from rag_desktop_app.models.conversation import Conversation


class Sidebar(QWidget):
    conversation_selected = Signal(str)

    COLLAPSED_WIDTH = 0          # 👈 IMPORTANT
    EXPANDED_WIDTH = 260

    def __init__(self):
        super().__init__()

        self.setObjectName("sidebar")

        self._expanded = False
        self._setup_ui()

        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        # Start fully collapsed
        self.setMinimumWidth(self.COLLAPSED_WIDTH)
        self.setMaximumWidth(self.COLLAPSED_WIDTH)

    def _setup_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(6)
        self.layout.setContentsMargins(0, 0, 0, 0)  # 👈 no margins when collapsed

        self._buttons = {}
        self.layout.addStretch()

    # --------------------------------------------------
    # COLLAPSE / EXPAND
    # --------------------------------------------------
    def toggle(self, expand: bool):
        self._expanded = expand

        # Restore margins only when expanded
        if expand:
            self.layout.setContentsMargins(8, 8, 8, 8)
        else:
            self.layout.setContentsMargins(0, 0, 0, 0)

        start = self.minimumWidth()
        end = self.EXPANDED_WIDTH if expand else self.COLLAPSED_WIDTH

        self.anim = QPropertyAnimation(self, b"minimumWidth")
        self.anim.setDuration(220)
        self.anim.setStartValue(start)
        self.anim.setEndValue(end)
        self.anim.start()

    # --------------------------------------------------
    # CONVERSATIONS
    # --------------------------------------------------
    def add_conversation(self, conversation: Conversation):
        if conversation.id in self._buttons:
            return

        btn = QPushButton(conversation.title)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setMinimumHeight(36)
        btn.setCheckable(True)

        btn.clicked.connect(
            lambda: self.conversation_selected.emit(conversation.id)
        )

        self.layout.insertWidget(0, btn)
        self._buttons[conversation.id] = btn

    def clear(self):
        for btn in self._buttons.values():
            btn.deleteLater()
        self._buttons.clear()
