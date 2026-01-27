from PySide6.QtCore import QThread, Signal
from services.rag_backend import RAGBackend


class RAGThread(QThread):
    """
    Background worker thread for calling the RAG backend API.
    Keeps the UI responsive while processing queries.
    """

    finished = Signal(dict)   # Emits full API response (JSON)
    error = Signal(str)

    def __init__(self, query: str):
        super().__init__()
        self.query = query
        self.backend = RAGBackend()

    def run(self):
        try:
            response = self.backend.process_query(self.query)
            # response is expected to be a dict:
            # {
            #   "query": "...",
            #   "answer": "...",
            #   "context_used": [...],
            #   "evaluation": {...},
            #   "pipeline": {...}
            # }
            self.finished.emit(response)

        except Exception as e:
            self.error.emit(str(e))
