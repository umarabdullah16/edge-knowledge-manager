from langchain_chroma import Chroma
from . import config

def create_and_store_embeddings(documents, embeddings):
    """
    Creates a new ChromaDB collection and stores the document embeddings.
    The database is persisted to disk.

    Args:
        documents (list): A list of LangChain Document objects.
        embeddings (HuggingFaceEmbeddings): The embedding model instance.
    """
    print("Storing embeddings in ChromaDB...")
    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=config.PERSIST_DIRECTORY,
        collection_name=config.COLLECTION_NAME
    )
    print("Embeddings stored and persisted successfully.")

def get_retriever(embeddings):
    """
    Initializes a retriever from an existing persistent ChromaDB.

    Args:
        embeddings (HuggingFaceEmbeddings): The embedding model instance.

    Returns:
        A LangChain retriever object.
    """
    print("Initializing retriever from existing ChromaDB...")
    vectorstore = Chroma(
        persist_directory=config.PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_name=config.COLLECTION_NAME
    )
    return vectorstore.as_retriever()
