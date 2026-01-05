import requests

class RAGBackend:
    def __init__(self):
        self.base_url = "http://localhost:8000"

    def process_query(self, query: str, file_path=None) -> str:
        print("🔥 FRONTEND → sending request to API")

        try:
            response = requests.post(
                f"{self.base_url}/query",
                json={"query": query},
                timeout=120
            )
            response.raise_for_status()

            data = response.json()
            return data.get("answer", "No answer returned from API")

        except requests.exceptions.ConnectionError:
            return "❌ Backend is not running."

        except requests.exceptions.Timeout:
            return "❌ Backend timed out."

        except Exception as e:
            return f"❌ API Error: {e}"
