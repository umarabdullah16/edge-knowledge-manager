import requests
from typing import Optional, Dict, Any


class RAGBackend:
    """
    Handles communication with the FastAPI RAG backend.
    UI and threads should never call requests directly.
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Query processing
    # ------------------------------------------------------------------
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Send a query to the backend and return the full structured response.

        Expected response format:
        {
            "query": "...",
            "answer": "...",
            "context_used": [...],
            "evaluation": {...},
            "pipeline": {...}
        }
        """
        url = f"{self.base_url}/query"
        payload = {"query": query}

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Backend query failed: {e}")

    # ------------------------------------------------------------------
    # Document ingestion
    # ------------------------------------------------------------------
    def ingest_document(self, file_path: str) -> None:
        """
        Upload a document to the backend for ingestion.
        """
        url = f"{self.base_url}/ingest"

        try:
            with open(file_path, "rb") as f:
                files = {"file": f}
                response = requests.post(url, files=files, timeout=120)

            response.raise_for_status()

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Document ingestion failed: {e}")
