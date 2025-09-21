"""
This module handles the generation of embeddings for text documents.
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from . import config

def get_embeddings():
    """
    Initializes and returns the Hugging Face embedding model.

    Returns:
        HuggingFaceEmbeddings: The embedding model instance.
    """
    device = "cpu"
    
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={'device': device}
    )
    return embeddings