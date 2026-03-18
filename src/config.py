
"""
This module contains the configuration settings for the Qdrant vector database project.
"""

# Embedding model configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/static-retrieval-mrl-en-v1"
EMBEDDING_DEVICE = "cpu"
# LLM model name (used by the RAG chain). Override as needed.
LLM_MODEL_NAME = "openai/gpt-oss-120b"
# Qdrant configuration
# --- Vector Database Configuration ---
PERSIST_DIRECTORY = "db"
COLLECTION_NAME = "my_documents"
# Retrieval configuration: how many top documents to retrieve for a query
# Reducing this decreases prompt/context size and token usage when calling the LLM
TOP_K = 5

# Retrieval mode: "vector" | "bm25" | "hybrid"
RETRIEVAL_MODE = "hybrid"

# Number of docs to return from BM25 retriever
BM25_K = 5

# Hybrid search weights [vector_weight, bm25_weight]
HYBRID_WEIGHTS = [0.7, 0.3]
