from langchain_chroma import Chroma
import chromadb
from src import config


# ======================================================
# INGEST
# ======================================================
def create_and_store_embeddings(documents, embeddings):
    """
    Creates or updates a persistent ChromaDB collection
    and stores document embeddings.
    """
    print("Storing embeddings in ChromaDB...")

    vectordb = Chroma(
        persist_directory=config.PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_name=config.COLLECTION_NAME
    )

    vectordb.add_documents(documents)
    vectordb.persist()

    print("Embeddings stored and persisted successfully.")


# ======================================================
# RETRIEVER
# ======================================================
def get_retriever(embeddings):
    """
    Initializes a retriever from the existing persistent ChromaDB.
    """
    print("Initializing retriever from existing ChromaDB...")

    vectordb = Chroma(
        persist_directory=config.PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_name=config.COLLECTION_NAME
    )

    search_kwargs = {"k": getattr(config, "TOP_K", 3)}
    return vectordb.as_retriever(search_kwargs=search_kwargs)


# ======================================================
# DOCUMENT STATISTICS  ✅ FIXED
# ======================================================
def get_document_statistics():
    """
    Retrieves statistics about documents and chunks stored in ChromaDB.
    """
    try:
        client = chromadb.PersistentClient(
            path=config.PERSIST_DIRECTORY
        )

        collection = client.get_collection(
            name=config.COLLECTION_NAME
        )

        data = collection.get(include=["metadatas"])

        metadatas = data.get("metadatas", []) or []

        doc_chunks = {}
        for meta in metadatas:
            source = meta.get("source", "unknown")
            filename = source.split("/")[-1] if source else "unknown"
            doc_chunks[filename] = doc_chunks.get(filename, 0) + 1

        return {
            "total_documents": len(doc_chunks),
            "total_chunks": len(metadatas),
            "documents": [
                {"filename": name, "chunks": count}
                for name, count in sorted(doc_chunks.items())
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
