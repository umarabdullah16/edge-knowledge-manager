"""
This module manages interactions with the Qdrant vector database.
"""

from qdrant_client import QdrantClient, models
from langchain_community.vectorstores import Qdrant
from . import config

def get_qdrant_client():
    """
    Initializes and returns the Qdrant client.

    Returns:
        QdrantClient: The Qdrant client instance.
    """
    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    return client

def create_and_store_embeddings(texts, embeddings):
    """
    Creates a Qdrant collection and stores the document embeddings.

    Args:
        texts (list): A list of document chunks.
        embeddings (HuggingFaceEmbeddings): The embedding model instance.
    """
    qdrant = Qdrant.from_documents(
        texts,
        embeddings,
        host=config.QDRANT_HOST,
        port=config.QDRANT_PORT,
        collection_name=config.COLLECTION_NAME,
    )
    return qdrant
