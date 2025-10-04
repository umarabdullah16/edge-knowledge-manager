
"""
This module contains the configuration settings for the Chroma vector database project.
"""
import os
# Embedding model configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/static-retrieval-mrl-en-v1"
EMBEDDING_DEVICE = "cpu"
# Chroma configuration
# --- Vector Database Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSIST_DIRECTORY = os.path.join(PROJECT_ROOT, "db")
COLLECTION_NAME = "my_documents"
