from PySide6.QtCore import QThread, Signal
from rag_desktop_app.services.rag_backend import RAGBackend


class RAGThread(QThread):
    """
    Background worker thread for calling the RAG backend API.

    Responsibilities:
    - Run backend queries off the UI thread
    - Emit structured, UI-safe data
    - Never contain business or UI logic
    """

    finished = Signal(dict)   # Emits UI-safe response dict
    error = Signal(str)

    def __init__(self, query: str):
        super().__init__()
        self.query = query
        self.backend = RAGBackend()

    def run(self):
        try:
            result = self.backend.process_query(self.query)
            self.finished.emit(result)

        except RuntimeError as e:
            # Clean backend-related error
            self.error.emit(str(e))

        except Exception as e:
            # Fallback: unexpected failure
            self.error.emit(f"Unexpected error: {e}")
