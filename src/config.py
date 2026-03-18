
"""
This module contains the configuration settings for the Qdrant vector database project.
"""

# Embedding model configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/static-retrieval-mrl-en-v1"
EMBEDDING_DEVICE = "cpu"
# LLM model name (used by the RAG chain). Override as needed.
LLM_MODEL_NAME = "openai/gpt-oss-120b"

# --- Agentic Tools Configuration ---
# Web search backend used by the tool-enabled RAG path.
WEB_SEARCH_BACKEND = "serper"

# Global default: enable web search for general/current-events questions.
WEB_SEARCH_ENABLED = True

# Serper API endpoint and request controls.
SERPER_API_URL = "https://google.serper.dev/search"
WEB_SEARCH_TIMEOUT_SECONDS = 8
WEB_SEARCH_MAX_RESULTS = 5
WEB_SEARCH_MAX_SNIPPET_CHARS = 240

# Math tool controls
MATH_TOOL_ENABLED = True
MATH_TOOL_MAX_EXPRESSION_CHARS = 120
MATH_TOOL_DECIMAL_PLACES = 10

# ReAct agent controls
REACT_MAX_STEPS = 4
REACT_TOOL_OUTPUT_CHARS = 2500

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

# MMR reranking controls (applies to vector retrieval in vector/hybrid modes)
# MMR improves result diversity by balancing relevance and novelty.
MMR_ENABLED = True

# Candidate pool size for MMR selection. Must be >= TOP_K.
MMR_FETCH_K = 20

# Relevance/diversity trade-off in [0.0, 1.0]
# 1.0 => prioritize relevance; 0.0 => prioritize diversity.
MMR_LAMBDA_MULT = 0.5
