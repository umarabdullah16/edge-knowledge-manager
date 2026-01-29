import requests
from typing import Dict, Any, List, Optional


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
        Send a query to the backend and return a UI-safe, structured response.

        UI-safe response format:
        {
            "answer": str,
            "sources": list,
            "evaluation": dict | None,
            "latency_ms": int | None,
            "raw": dict        # full backend response (debug / optional)
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
                "sources": backend_data.get("context_used", []),
                "evaluation": backend_data.get("evaluation"),
                "latency_ms": backend_data.get("pipeline", {}).get("latency_ms"),
                "raw": backend_data,  # useful for debugging / future UI panels
            }

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Backend query failed: {e}")

        except ValueError:
            # JSON decoding error
            raise RuntimeError("Backend returned invalid JSON")

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

        except FileNotFoundError:
            raise RuntimeError(f"File not found: {file_path}")
