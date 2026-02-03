import requests
from typing import Dict, Any


class RAGBackend:
    """
    Handles all communication with the FastAPI RAG backend.

    Rules:
    - UI must NEVER call requests directly
    - Threads must NEVER parse backend JSON
    - This class is the ONLY place that understands backend structure
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Query processing
    # ------------------------------------------------------------------
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Send a query to the backend and return a UI-safe response.

        UI-safe response format (CURRENT BACKEND):
        {
            "answer": str,
            "raw": dict
        }
        """

        url = f"{self.base_url}/query"
        payload = {"query": query}

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()

            backend_data = response.json()

            return {
                "answer": backend_data.get("answer", ""),
                "raw": backend_data,  # keep raw for debugging / future expansion
            }

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Backend query failed: {e}")

        except ValueError:
            raise RuntimeError("Backend returned invalid JSON")

    # ------------------------------------------------------------------
    # Document ingestion
    # ------------------------------------------------------------------
    def ingest_document(self, file_path: str) -> Dict[str, Any]:
        """
        Upload a document to the backend for ingestion.

        Returns:
        {
            "filename": str,
            "chunks_processed": int
        }
        """

        url = f"{self.base_url}/ingest"

        try:
            with open(file_path, "rb") as f:
                files = {"file": f}
                response = requests.post(url, files=files, timeout=120)

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Document ingestion failed: {e}")

        except FileNotFoundError:
            raise RuntimeError(f"File not found: {file_path}")

    # ------------------------------------------------------------------
    # Document statistics
    # ------------------------------------------------------------------
    def get_document_statistics(self) -> Dict[str, Any]:
        """
        Fetch statistics about indexed documents.
        """

        url = f"{self.base_url}/documents/statistics"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to fetch document statistics: {e}")

    # ------------------------------------------------------------------
    # Prompt preview (debug / optional)
    # ------------------------------------------------------------------
    def preview_prompt(self, query: str) -> Dict[str, Any]:
        """
        Fetch the assembled prompt and retrieved context for debugging.
        """

        url = f"{self.base_url}/preview_prompt"
        payload = {"query": query}

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Prompt preview failed: {e}")
