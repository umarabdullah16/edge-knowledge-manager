import sys
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QPushButton,
    QLabel
)
from PySide6.QtCore import QThread, Signal

# ✅ IMPORT REAL BACKEND (IMPORTANT)
from rag_backend import RAGBackend


# ==============================
# Background thread for API call
# ==============================
class RAGThread(QThread):
    response_ready = Signal(str)

    def __init__(self, query: str):
        super().__init__()
        self.query = query
        self.backend = RAGBackend()

    def run(self):
        answer = self.backend.process_query(self.query)
        self.response_ready.emit(answer)


# ==============================
# Main UI
# ==============================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Knowledge Management System")

        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout()

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)

        self.input_box = QTextEdit()
        self.input_box.setFixedHeight(60)

        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self.send_query)

        layout.addWidget(QLabel("Chat"))
        layout.addWidget(self.chat_display)
        layout.addWidget(self.input_box)
        layout.addWidget(send_btn)

        central.setLayout(layout)

    def send_query(self):
        query = self.input_box.toPlainText().strip()
        if not query:
            return

        self.chat_display.append(f"🧑 You: {query}")
        self.input_box.clear()

        self.thread = RAGThread(query)
        self.thread.response_ready.connect(self.display_response)
        self.thread.start()

    def display_response(self, response):
        self.chat_display.append(f"🤖 AI: {response}")


# ==============================
# App Entry Point
# ==============================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
