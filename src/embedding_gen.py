"""
This module handles the generation of embeddings for text documents.
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
import config

def get_embeddings():
    """
    Initializes and returns the Hugging Face embedding model.

    Returns:
        HuggingFaceEmbeddings: The embedding model instance.
    """
    print("Initializing embedding model...")
    # Specify the device to use for embeddings (e.g., 'cpu' or 'cuda')
    model_kwargs = {"device": config.EMBEDDING_DEVICE}
    
    # Initialize the HuggingFace embeddings model using the updated library
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
    )
    return embeddings
