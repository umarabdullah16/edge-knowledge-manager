from fastapi.testclient import TestClient
import io
import api
from langchain.schema import Document


def test_ingest_endpoint(monkeypatch):
    # Patch the loader to avoid reading the actual uploaded file
    def fake_load_and_split(path):
        return [Document(page_content="hello world", metadata={"page": 0, "chunk_id": "c1", "start_char": 0, "end_char": 11})]

    monkeypatch.setattr(api.doc_process, "load_and_split_pdf", fake_load_and_split)

    # Patch embedding/model and vectorstore to no-op
    monkeypatch.setattr(api.embedding_gen, "get_embeddings", lambda: object())
    monkeypatch.setattr(api.vectorstore_manager, "create_and_store_embeddings", lambda docs, emb: None)

    client = TestClient(api.app)

    files = {"file": ("test.pdf", b"%PDF-1.4 fake pdf content", "application/pdf")}
    resp = client.post("/ingest", files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert body["message"] == "File ingested successfully"
    assert body["chunks_processed"] == 1


def test_query_endpoint(monkeypatch):
    # Patch embedding/model and RAG chain
    monkeypatch.setattr(api.embedding_gen, "get_embeddings", lambda: object())

    class FakeChain:
        def invoke(self, q):
            return "fake answer"

    monkeypatch.setattr(api.rag_processor, "setup_rag_chain", lambda emb: FakeChain())

    client = TestClient(api.app)
    resp = client.post("/query", json={"query": "Hello"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "Hello"
    assert body["answer"] == "fake answer"
