
"""
This module contains the configuration settings for the Qdrant vector database project.
"""

# Embedding model configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/static-retrieval-mrl-en-v1"
EMBEDDING_DEVICE = "cpu"
# Qdrant configuration
# --- Vector Database Configuration ---
PERSIST_DIRECTORY = "db"
COLLECTION_NAME = "my_documents"
