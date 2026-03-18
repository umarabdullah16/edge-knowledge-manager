
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
- FastAPI endpoints for ingestion, query, and document statistics.
- Unit and functional tests with CI on push/PR to `main`.

# 🛠 Tech Stack

Vector Database: ChromaDB (Local persistence)

Embeddings: sentence-transformers/static-retrieval-mrl-en-v1

LLM: Llama3-70b-8192 via Groq API

Orchestration: LangChain

Evaluation: Ragas

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
```

# 💻 Usage

1. Ingest documents

Parse PDFs and create embeddings. Data is saved to the `db/` folder automatically.

Ingest a single file:
```
python -m src.main --pdf path/to/document.pdf
```

Start API server:
```
python -m api
```

Query via API (example):
```
POST /query
{
	"query": "What are the main conclusions of the document?"
}
```

2. Ask questions (CLI)

Query your local knowledge base.
```
python -m qna --query "What are the main conclusions of the document?"
```

3. Evaluate performance

Run a Ragas evaluation to test the retrieval and generation quality of your specific documents. This generates synthetic questions based on your PDF and scores the system.
```
python -m src.evaluate_rag --pdf path/to/document.pdf
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
├── db/                 # Created automatically; stores vector data
├── docs/               # Project and testing documentation
├── src/                # Core logic
│   ├── main.py         # Main entry point for ingestion
│   ├── evaluate_rag.py # RAG evaluation pipeline
│   └── ...
├── api.py              # FastAPI service
├── qna.py              # Query CLI entry
├── test_*.py           # Unit and functional tests
├── .github/workflows/  # CI pipelines
├── requirements.txt    # Dependencies
├── README.md           # Documentation
└── .env                # API Keys (Excluded from git)
```

# 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

# 📄 License

MIT
