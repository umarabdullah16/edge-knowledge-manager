from fastapi.testclient import TestClient
import io
import api
try:
    from langchain_core.documents import Document
except ImportError:  # Backward compatibility with older LangChain
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

    monkeypatch.setattr(
        api.rag_processor,
        "setup_rag_chain",
        lambda emb: FakeChain(),
    )

    client = TestClient(api.app)
    resp = client.post("/query", json={"query": "Hello"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "Hello"
    assert body["answer"] == "fake answer"


def test_query_endpoint_ignores_tool_toggles_in_payload(monkeypatch):
    monkeypatch.setattr(api.embedding_gen, "get_embeddings", lambda: object())

    class FakeChain:
        def invoke(self, q):
            return "ok"

    monkeypatch.setattr(api.rag_processor, "setup_rag_chain", lambda emb: FakeChain())

    client = TestClient(api.app)
    resp = client.post("/query", json={"query": "2 + 2", "use_math_tool": True, "use_web_search": False})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "ok"


def test_documents_statistics_endpoint(monkeypatch):
    """Test the /documents/statistics endpoint"""
    # Patch the vectorstore_manager.get_document_statistics function
    def fake_statistics():
        return {
            "total_documents": 2,
            "total_chunks": 100,
            "documents": [
                {"name": "doc1.pdf", "chunks": 50},
                {"name": "doc2.pdf", "chunks": 50}
            ]
        }

    monkeypatch.setattr(api.vectorstore_manager, "get_document_statistics", fake_statistics)

    client = TestClient(api.app)
    resp = client.get("/documents/statistics")
    assert resp.status_code == 200
    body = resp.json()
    
    # Verify response structure
    assert "total_documents" in body
    assert "total_chunks" in body
    assert "documents" in body
    
    # Verify values
    assert body["total_documents"] == 2
    assert body["total_chunks"] == 100
    assert len(body["documents"]) == 2
    
    # Verify document structure
    assert body["documents"][0]["name"] == "doc1.pdf"
    assert body["documents"][0]["chunks"] == 50
    assert body["documents"][1]["name"] == "doc2.pdf"
    assert body["documents"][1]["chunks"] == 50
