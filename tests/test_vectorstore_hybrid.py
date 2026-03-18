from src import vectorstore_manager, config


class FakeChroma:
    def __init__(self, *args, **kwargs):
        self._vector_retriever = {"kind": "vector", "k": kwargs.get("k")}

    def as_retriever(self, search_kwargs=None, search_type=None):
        return {
            "kind": "vector",
            "k": (search_kwargs or {}).get("k"),
            "fetch_k": (search_kwargs or {}).get("fetch_k"),
            "lambda_mult": (search_kwargs or {}).get("lambda_mult"),
            "search_type": search_type,
        }

    def get(self, include=None):
        return {
            "documents": ["alpha beta", "gamma delta"],
            "metadatas": [{"source": "a.pdf"}, {"source": "b.pdf"}],
        }


class FakeBM25Retriever:
    def __init__(self):
        self.k = None


class FakeEnsembleRetriever:
    def __init__(self, retrievers, weights):
        self.retrievers = retrievers
        self.weights = weights


def test_get_retriever_vector_mode(monkeypatch):
    monkeypatch.setattr(vectorstore_manager, "Chroma", FakeChroma)
    monkeypatch.setattr(config, "RETRIEVAL_MODE", "vector")
    monkeypatch.setattr(config, "TOP_K", 4)
    monkeypatch.setattr(config, "MMR_ENABLED", False)

    retriever = vectorstore_manager.get_retriever(object())

    assert retriever["kind"] == "vector"
    assert retriever["k"] == 4
    assert retriever["search_type"] is None


def test_get_retriever_vector_mode_with_mmr(monkeypatch):
    monkeypatch.setattr(vectorstore_manager, "Chroma", FakeChroma)
    monkeypatch.setattr(config, "RETRIEVAL_MODE", "vector")
    monkeypatch.setattr(config, "TOP_K", 4)
    monkeypatch.setattr(config, "MMR_ENABLED", True)
    monkeypatch.setattr(config, "MMR_FETCH_K", 12)
    monkeypatch.setattr(config, "MMR_LAMBDA_MULT", 0.2)

    retriever = vectorstore_manager.get_retriever(object())

    assert retriever["kind"] == "vector"
    assert retriever["search_type"] == "mmr"
    assert retriever["k"] == 4
    assert retriever["fetch_k"] == 12
    assert retriever["lambda_mult"] == 0.2


def test_get_retriever_bm25_mode(monkeypatch):
    monkeypatch.setattr(vectorstore_manager, "Chroma", FakeChroma)
    monkeypatch.setattr(config, "RETRIEVAL_MODE", "bm25")
    monkeypatch.setattr(config, "TOP_K", 3)
    monkeypatch.setattr(config, "BM25_K", 2)
    monkeypatch.setattr(config, "MMR_ENABLED", False)

    fake_bm25 = FakeBM25Retriever()

    class FakeBM25Factory:
        @staticmethod
        def from_documents(docs):
            assert len(docs) == 2
            return fake_bm25

    monkeypatch.setattr(vectorstore_manager, "BM25Retriever", FakeBM25Factory)

    retriever = vectorstore_manager.get_retriever(object())

    assert retriever is fake_bm25
    assert retriever.k == 2


def test_get_retriever_hybrid_mode(monkeypatch):
    monkeypatch.setattr(vectorstore_manager, "Chroma", FakeChroma)
    monkeypatch.setattr(config, "RETRIEVAL_MODE", "hybrid")
    monkeypatch.setattr(config, "TOP_K", 5)
    monkeypatch.setattr(config, "BM25_K", 5)
    monkeypatch.setattr(config, "HYBRID_WEIGHTS", [0.6, 0.4])
    monkeypatch.setattr(config, "MMR_ENABLED", False)

    class FakeBM25Factory:
        @staticmethod
        def from_documents(docs):
            retriever = FakeBM25Retriever()
            retriever.k = 5
            return retriever

    monkeypatch.setattr(vectorstore_manager, "BM25Retriever", FakeBM25Factory)
    monkeypatch.setattr(vectorstore_manager, "EnsembleRetriever", FakeEnsembleRetriever)

    retriever = vectorstore_manager.get_retriever(object())

    assert isinstance(retriever, FakeEnsembleRetriever)
    assert len(retriever.retrievers) == 2
    assert retriever.retrievers[0]["kind"] == "vector"
    assert retriever.retrievers[0]["search_type"] is None
    assert retriever.weights == [0.6, 0.4]
