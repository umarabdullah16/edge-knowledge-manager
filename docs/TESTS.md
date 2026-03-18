# Tests

This project includes unit and functional tests. The functional tests mock heavy dependencies (models and vector DB) so they run quickly in CI or locally.

Running tests

1. Activate the virtual environment:

```bash
source venv/bin/activate
```

2. Install test requirements:

```bash
pip install -r requirements.txt
pip install pytest
```

3. Run tests:

```bash
pytest -q
```

Notes

- `test_doc_process.py` validates the metadata-augmented chunker by mocking `PyPDFLoader`.
- `test_api.py` performs functional API tests using FastAPI's `TestClient` and monkeypatches the heavy components so tests don't require a GPU, Groq key, or running Qdrant/ChromaDB.
