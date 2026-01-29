from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QScrollArea, QPushButton, QFrame, QLineEdit
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
import qtawesome as qta

from rag_desktop_app.ui.message_bubble import MessageBubble


class ChatArea(QWidget):
    send_message = Signal(str)

    # Window control signals
    minimize_requested = Signal()
    maximize_requested = Signal()
    close_requested = Signal()

    def __init__(self):
        super().__init__()
        self._loading_label = None
        self._setup_ui()

    # ======================================================
    # UI SETUP
    # ======================================================
    def _setup_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # ==================================================
        # HEADER (TOP BAR)
        # ==================================================
        self.chat_header = QWidget()
        self.chat_header.setObjectName("chatHeader")
        self.chat_header.setFixedHeight(56)

        header_layout = QHBoxLayout(self.chat_header)
        header_layout.setContentsMargins(16, 0, 10, 0)
        header_layout.setSpacing(12)

        # -------- Left side --------
        self.menu_button = QPushButton()
        self.menu_button.setIcon(qta.icon("fa5s.bars"))
        self.menu_button.setFlat(True)
        self.menu_button.setCursor(Qt.PointingHandCursor)

        title = QLabel("RAG Assistant")
        title.setFont(QFont("Segoe UI", 14, QFont.Medium))

        header_layout.addWidget(self.menu_button)
        header_layout.addWidget(title)
        header_layout.addStretch()

        # -------- Right side (window controls) --------
        controls = QWidget()
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(6)

        btn_min = QPushButton()
        btn_min.setIcon(qta.icon("fa5s.minus"))

        btn_max = QPushButton()
        btn_max.setIcon(qta.icon("fa5s.square"))

        btn_close = QPushButton()
        btn_close.setIcon(qta.icon("fa5s.times"))

        for btn in (btn_min, btn_max, btn_close):
            btn.setObjectName("windowControl")
            btn.setFixedSize(32, 28)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setFlat(True)

        btn_min.clicked.connect(self.minimize_requested.emit)
        btn_max.clicked.connect(self.maximize_requested.emit)
        btn_close.clicked.connect(self.close_requested.emit)

        controls_layout.addWidget(btn_min)
        controls_layout.addWidget(btn_max)
        controls_layout.addWidget(btn_close)

        header_layout.addWidget(controls)

        self.main_layout.addWidget(self.chat_header)

        # ==================================================
        # START STATE
        # ==================================================
        self.start_container = QWidget()
        start_layout = QVBoxLayout(self.start_container)
        start_layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        start_layout.setContentsMargins(0, 110, 0, 0)
        start_layout.setSpacing(24)

        greeting = QLabel("Hello")
        greeting.setFont(QFont("Segoe UI", 30, QFont.DemiBold))
        greeting.setAlignment(Qt.AlignCenter)

        subtitle = QLabel("Ask questions about your documents")
        subtitle.setFont(QFont("Segoe UI", 14))
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("opacity: 0.7;")

        start_layout.addWidget(greeting)
        start_layout.addWidget(subtitle)

        # -------------------------------
        # SUGGESTION CARDS
        # -------------------------------
        cards_row = QHBoxLayout()
        cards_row.setSpacing(18)

        suggestions = [
            ("Summarize document", "fa5s.align-left"),
            ("Explain a concept", "fa5s.code"),
            ("Search information", "fa5s.search"),
            ("Extract key points", "fa5s.list"),
        ]

        for text, icon in suggestions:
            card = QFrame()
            card.setObjectName("suggestionCard")
            card.setFixedSize(180, 72)
            card.setCursor(Qt.PointingHandCursor)

            cl = QHBoxLayout(card)
            cl.setContentsMargins(16, 12, 16, 12)
            cl.setSpacing(12)

            icon_lbl = QLabel()
            icon_lbl.setPixmap(qta.icon(icon).pixmap(18, 18))

            txt_lbl = QLabel(text)
            txt_lbl.setFont(QFont("Segoe UI", 11, QFont.Medium))
            txt_lbl.setWordWrap(True)

            cl.addWidget(icon_lbl)
            cl.addWidget(txt_lbl)

            card.mousePressEvent = lambda e, t=text: self._send_from_start(t)
            cards_row.addWidget(card)

        start_layout.addLayout(cards_row)

        # -------------------------------
        # START INPUT
        # -------------------------------
        self.start_input = self._build_input_pill(
            "Ask about these documents…", wide=False
        )
        start_layout.addWidget(self.start_input, alignment=Qt.AlignHCenter)

        self.main_layout.addWidget(self.start_container, 1)

        # ==================================================
        # CHAT STATE
        # ==================================================
        self.messages_area = QScrollArea()
        self.messages_area.setObjectName("messagesArea")
        self.messages_area.setWidgetResizable(True)
        self.messages_area.hide()

        msg_container = QWidget()
        self._msg_layout = QVBoxLayout(msg_container)
        self._msg_layout.setSpacing(14)
        self._msg_layout.addStretch()

        self.messages_area.setWidget(msg_container)
        self.main_layout.addWidget(self.messages_area, 1)

        # -------------------------------
        # CHAT INPUT
        # -------------------------------
        self.chat_input = self._build_input_pill(
            "Type your message…", wide=True
        )
        self.chat_input.hide()

        self.main_layout.addSpacing(18)
        self.main_layout.addWidget(self.chat_input, alignment=Qt.AlignHCenter)
        self.main_layout.addSpacing(20)

    # ======================================================
    # INPUT PILL (FOCUS HIGHLIGHT)
    # ======================================================
    def _build_input_pill(self, placeholder, wide):
        pill = QFrame()
        pill.setObjectName("inputPill")
        pill.setAttribute(Qt.WA_StyledBackground, True)

        pill.setFixedSize(
            560 if wide else 520,
            58 if wide else 52
        )

        layout = QHBoxLayout(pill)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(12)

        attach = QPushButton()
        attach.setObjectName("pillIcon")
        attach.setIcon(qta.icon("fa5s.paperclip"))
        attach.setFlat(True)

        field = QLineEdit()
        field.setObjectName("pillField")
        field.setPlaceholderText(placeholder)

        send = QPushButton()
        send.setObjectName("pillIcon")
        send.setIcon(qta.icon("fa5s.paper-plane"))
        send.setCursor(Qt.PointingHandCursor)

        layout.addWidget(attach)
        layout.addWidget(field, 1)
        layout.addWidget(send)

        # -------- Focus handling --------
        def _refresh_style():
            pill.style().unpolish(pill)
            pill.style().polish(pill)
            pill.update()

        def focus_in(e):
            pill.setProperty("focused", True)
            _refresh_style()
            QLineEdit.focusInEvent(field, e)

        def focus_out(e):
            pill.setProperty("focused", False)
            _refresh_style()
            QLineEdit.focusOutEvent(field, e)

        field.focusInEvent = focus_in
        field.focusOutEvent = focus_out
        # --------------------------------

        field.returnPressed.connect(lambda: self._emit_message(field.text()))
        send.clicked.connect(lambda: self._emit_message(field.text()))

        pill.input_field = field
        return pill

    # ======================================================
    # STATE SWITCHING
    # ======================================================
    def switch_to_chat(self):
        self.start_container.hide()
        self.start_input.hide()
        self.messages_area.show()
        self.chat_input.show()
        self.chat_input.input_field.setFocus()

    def reset_to_start(self):
        self.messages_area.hide()
        self.chat_input.hide()
        self.start_container.show()
        self.start_input.show()

    # ======================================================
    # MESSAGE EMISSION
    # ======================================================
    def _emit_message(self, text: str):
        text = text.strip()
        if not text:
            return
        self.switch_to_chat()
        self.send_message.emit(text)

    def _send_from_start(self, text: str):
        self.switch_to_chat()
        self.send_message.emit(text)

    # ======================================================
    # PUBLIC UI API
    # ======================================================
    def add_user_message(self, text: str):
        bubble = MessageBubble(message=text, is_user=True)
        self._insert_message(bubble, align_right=True)

    def add_bot_message(self, text: str):
        bubble = MessageBubble(message=text, is_user=False)
        self._insert_message(bubble, align_right=False)

    def add_system_message(self, text: str):
        label = QLabel(text)
        label.setAlignment(Qt.AlignCenter)
        label.setObjectName("systemMessage")
        self._msg_layout.insertWidget(self._msg_layout.count() - 1, label)
        self._scroll_to_bottom()

    def show_loading(self):
        if self._loading_label is None:
            self._loading_label = QLabel("Thinking…")
            self._loading_label.setObjectName("loadingMessage")
            self._msg_layout.insertWidget(
                self._msg_layout.count() - 1,
                self._loading_label
            )
            self._scroll_to_bottom()

    def hide_loading(self):
        if self._loading_label is not None:
            try:
                self._loading_label.setParent(None)
                self._loading_label.deleteLater()
            except RuntimeError:
                pass
            finally:
                self._loading_label = None

    # ======================================================
    # INTERNAL HELPERS
    # ======================================================
    def _insert_message(self, widget: QWidget, align_right: bool):
        row = QHBoxLayout()
        row.setContentsMargins(12, 0, 12, 0)

        if align_right:
            row.addStretch()
            row.addWidget(widget)
        else:
            row.addWidget(widget)
            row.addStretch()

        container = QWidget()
        container.setLayout(row)

        self._msg_layout.insertWidget(
            self._msg_layout.count() - 1,
            container
        )
        self._scroll_to_bottom()

    def _scroll_to_bottom(self):
        bar = self.messages_area.verticalScrollBar()
        bar.setValue(bar.maximum())
