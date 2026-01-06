import requests

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
