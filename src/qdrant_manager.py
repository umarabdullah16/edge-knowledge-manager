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
        force_recreate=False,
    )
    return qdrant

def search_documents(query_text, embeddings, limit=5):
    """
    Searches for relevant documents in the Qdrant collection.

    Args:
        query_text (str): The text to search for.
        embeddings: The embedding model to use.
        limit (int): The maximum number of results to return.

    Returns:
        list: A list of search result hits.
    """
    # Embed the query
    query_vector = embeddings.embed_query(query_text)

    # Initialize the Qdrant client
    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    
    # Perform the search
    hits = client.search(
        collection_name=config.COLLECTION_NAME,
        query_vector=query_vector,
        limit=limit
    )
    
    return hits