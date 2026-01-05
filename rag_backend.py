import requests

class RAGBackend:
    def __init__(self):
        # Change IP if backend is on another machine
        self.base_url = "http://localhost:8000"

    def process_query(self, query: str, file_path=None) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/query",
                json={"query": query},
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            return data["answer"]

        except requests.exceptions.RequestException as e:
            return f"API Error: {e}"
