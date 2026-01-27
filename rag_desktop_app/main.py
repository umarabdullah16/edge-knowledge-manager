import sys
from PySide6.QtWidgets import QApplication

from rag_desktop_app.ui.main_window import MainWindow


def load_styles(app: QApplication):
    """
    Load global QSS styles.
    """
    try:
        with open("rag_desktop_app/styles/dark_theme.qss", "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())
    except Exception as e:
        print(f"[Style] Failed to load stylesheet: {e}")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("RAG Assistant")
    app.setApplicationVersion("1.0")

    load_styles(app)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
