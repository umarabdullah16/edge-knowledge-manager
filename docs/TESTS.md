# Tests

This project includes unit and functional tests. The functional tests mock heavy dependencies (models and vector DB) so they run quickly in CI or locally.

## CI pipeline

Basic tests are automatically run by GitHub Actions using:

- `.github/workflows/basic-tests.yml`

Triggers:

- push to `main`
- pull request to `main`

The CI job installs dependencies from `requirements.txt` and executes:

```bash
pytest -q
```

## Running tests locally

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
- `test_api.py` performs functional API tests using FastAPI's `TestClient` and monkeypatches heavy components so tests don't require a GPU, Groq key, or an external vector DB service.
