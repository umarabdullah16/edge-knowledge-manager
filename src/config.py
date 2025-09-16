
"""
This module contains the configuration settings for the Qdrant vector database project.
"""

# Embedding model configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/static-retrieval-mrl-en-v1"

# Qdrant configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "my_document_collection"
