import requests
import os

class RAGBackend:
    def __init__(self):
        self.base_url = "http://localhost:8000"

    def process_query(self, query: str, file_path=None) -> str:
        print("FRONTEND → API CALLED")

        try:
            response = requests.post(
                f"{self.base_url}/query",
                json={"query": query},
                timeout=120
            )
            response.raise_for_status()
            return response.json().get("answer", "No answer returned")

        except requests.exceptions.ConnectionError:
            return "❌ Backend is not running."

        except requests.exceptions.Timeout:
            return "❌ Backend timeout."

        except Exception as e:
            return f"❌ API Error: {e}"

    def ingest_document(self, pdf_path: str):
        """
        Uploads a PDF file to the FastAPI /ingest endpoint.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")

        try:
            with open(pdf_path, "rb") as f:
                files = {
                    "file": (os.path.basename(pdf_path), f, "application/pdf")
                }
                response = requests.post(
                    f"{self.base_url}/ingest",
                    files=files,
                    timeout=300
                )

            response.raise_for_status()
            return response.json()

        except requests.exceptions.ConnectionError:
            raise Exception("❌ Backend is not running.")

        except requests.exceptions.Timeout:
            raise Exception("❌ Ingestion timeout.")

        except Exception as e:
            raise Exception(f"❌ Ingestion failed: {e}")
