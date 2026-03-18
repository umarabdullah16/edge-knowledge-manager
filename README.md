
# Edge Knowledge Manager

A local-first Retrieval-Augmented Generation (RAG) system optimized for Raspberry Pi 5 and macOS.

Manage, ingest, and query private PDF documents using a lightweight architecture without requiring heavy containerized vector database services.

# 📖 Project Overview

The Edge Knowledge Manager allows users to build a personal knowledge base from private documents (PDFs) and query them using natural language. While many RAG systems require powerful servers or cloud vector databases, this project was built with a specific goal: to run efficiently on low-power edge devices.

# Architecture Decisions

This project was explicitly designed to overcome the limitations of running complex RAG pipelines on low-power hardware. While standard approaches often rely on heavy, containerized vector databases, this system utilizes ChromaDB in an embedded mode.

This design choice removes the need for Docker or external server processes, significantly reducing memory overhead and ensuring native compatibility with the ARM64 architecture found on devices like the Raspberry Pi 5.

# 🚀 Features

- Edge-optimized: tested on Raspberry Pi 5 and macOS.
- No Docker required: runs fully in a Python virtual environment with embedded Chroma persistence.
- High-performance inference: Groq-hosted LLM responses with local retrieval.
- PDF ingestion pipeline with metadata-augmented chunking.
- MMR reranking support for runtime retrieval (API/CLI/evaluation), not just offline evaluation.
- FastAPI endpoints for ingestion, query, and document statistics.
- Unit and functional tests with CI on push/PR to `main`.

# 🛠 Tech Stack

Vector Database: ChromaDB (Local persistence)

Embeddings: sentence-transformers/static-retrieval-mrl-en-v1

LLM: Llama3-70b-8192 via Groq API

Orchestration: LangChain

Evaluation: Ragas

Retrieval: Chroma vector search + BM25 + hybrid + MMR reranking

# ⚙️ Retrieval Configuration

Retrieval behavior is configured in [src/config.py](src/config.py).

- `RETRIEVAL_MODE`: `vector` | `bm25` | `hybrid`
- `TOP_K`: final number of chunks returned to prompt context
- `BM25_K`: BM25 retrieval depth when lexical retrieval is used
- `HYBRID_WEIGHTS`: `[vector_weight, bm25_weight]` for hybrid fusion
- `MMR_ENABLED`: enable Max Marginal Relevance (diversity-aware reranking) for vector retrieval
- `MMR_FETCH_K`: candidate pool size used by MMR (should be `>= TOP_K`)
- `MMR_LAMBDA_MULT`: relevance/diversity tradeoff in `[0.0, 1.0]`

When `MMR_ENABLED = True`, MMR applies automatically to:

- vector mode
- vector side of hybrid mode
- all runtime paths using the shared retriever (`qna.py`, `api.py`, and evaluation)

# 📦 Installation

You can set this up on a standard machine (Mac/Linux/Windows) or a Raspberry Pi.

Prerequisites

- Python 3.10+
- A Groq API key

1. Clone the repository
```
git clone https://github.com/umarabdullah16/edge-knowledge-manager.git
cd edge-knowledge-manager
```

2. Environment setup

Option A: Standard Machine (Mac/Linux/Windows)
```
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

Option B: Raspberry Pi 5 (ARM64)

The Pi requires specific attention to PyTorch installation due to the ARM architecture.
```
 1. Update system
sudo apt update && sudo apt install python3-venv python3-pip -y

 2. Create and activate venv
python3 -m venv venv
source venv/bin/activate

 3. Install PyTorch specifically for CPU/ARM
pip install torch --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)

 4. Install remaining dependencies
pip install -r requirements.txt
```

3. Configuration

Create a .env file in the root directory and add your API key:
```
GROQ_API_KEY=gsk_your_actual_api_key_here
SERPER_API_KEY=your_serper_api_key_here
```

`SERPER_API_KEY` is only required when web search augmentation is enabled.
If your environment already uses `SERPER_KEY`, that is also supported.

# 💻 Usage

1. Ingest documents

Parse PDFs and create embeddings. Data is saved to the `db/` folder automatically.

Ingest a single file:
```
python -m src.main --pdf data/pdfs/your-document.pdf
```

Start API server:
```
python -m api
```

Query via API (example):
```
POST /query
{
	"query": "What are the main conclusions of the document?",
	"use_web_search": true,
	"use_math_tool": true
}
```

2. Ask questions (CLI)

Query your local knowledge base.
```
python -m qna --query "What are the main conclusions of the document?"

# Enable Serper web augmentation for this run
python -m qna --query "Latest updates related to this topic" --use-web-search

# Disable math tool for a run (enabled by default)
python -m qna --query "What is 125 * 36?" --disable-math-tool
```

Math tool notes:

- The math tool is enabled by default and safely evaluates arithmetic/scientific expressions.
- Supported examples: `2 + 2`, `10/4`, `sqrt(16) + pi`, `log10(1000)`.

3. Evaluate performance

Run a Ragas evaluation to test the retrieval and generation quality of your specific documents. This generates synthetic questions based on your PDF and scores the system.
```
python -m src.evaluate_rag --queries data/squad_queries.jsonl --ground_truth data/squad_ground_truth.jsonl --top_k 5
```

Optional runtime overrides for evaluation runs:

```
python -m src.evaluate_rag \
	--queries data/squad_queries.jsonl \
	--ground_truth data/squad_ground_truth.jsonl \
	--retrieval_mode hybrid \
	--top_k 5 \
	--bm25_k 5 \
	--hybrid_weights 0.7,0.3
```

Prepare SQuAD-based evaluation files and ingest sample corpus:

```
python -m src.ingest_squad --n_samples 100
```

# ✅ Continuous Integration

This repository includes a GitHub Actions workflow that runs the basic test suite on:

- push to `main`
- pull requests targeting `main`

Workflow file:

- `.github/workflows/basic-tests.yml`

The workflow sets up Python, installs dependencies from `requirements.txt`, and runs:

```
pytest -q
```

# 📂 Project Structure
```
edge-knowledge-manager/
├── data/pdfs/          # Source/sample PDFs used for ingestion
├── db/                 # Created automatically; stores vector data
├── docs/               # Project and testing documentation
├── src/                # Core logic
│   ├── main.py         # Main entry point for ingestion
│   ├── evaluate_rag.py # RAG evaluation pipeline
│   └── ...
├── api.py              # FastAPI service
├── qna.py              # Query CLI entry
├── tests/              # Unit and functional tests
├── pytest.ini          # Test discovery configuration
├── .github/workflows/  # CI pipelines
├── requirements.txt    # Dependencies
├── README.md           # Documentation
└── .env                # API Keys (Excluded from git)
```

# 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

# 📄 License

MIT
