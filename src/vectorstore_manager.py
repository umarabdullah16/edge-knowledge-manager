from langchain_chroma import Chroma
from src import config

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
    # Limit number of retrieved documents to avoid creating overly large prompts
    search_kwargs = {"k": getattr(config, "TOP_K", 3)}
    return vectorstore.as_retriever(search_kwargs=search_kwargs)

def get_document_statistics():
    """
    Retrieves statistics about documents and chunks stored in ChromaDB.
    
    Returns:
        dict: Contains total_documents, total_chunks, and a list of documents
              with their chunk counts.
    """
    try:
        # Connect to ChromaDB without embeddings
        client = __import__('chromadb').PersistentClient(path=config.PERSIST_DIRECTORY)
        collection = client.get_collection(config.COLLECTION_NAME)
        
        # Get all documents
        all_data = collection.get(include=['metadatas'])
        
        # Count chunks by document source
        doc_chunks = {}
        for metadata in all_data.get('metadatas', []):
            source = metadata.get('source', 'unknown')
            # Extract just the filename
            filename = source.split('/')[-1] if source else 'unknown'
            doc_chunks[filename] = doc_chunks.get(filename, 0) + 1
        
        # Sort by filename
        sorted_docs = sorted(doc_chunks.items())
        
        return {
            "total_documents": len(sorted_docs),
            "total_chunks": len(all_data.get('ids', [])),
            "documents": [
                {"name": name, "chunks": count} 
                for name, count in sorted_docs
            ]
        }
    except Exception as e:
        print(f"❌ Error retrieving document statistics: {e}")
        return {
            "total_documents": 0,
            "total_chunks": 0,
            "documents": [],
            "error": str(e)
        }
