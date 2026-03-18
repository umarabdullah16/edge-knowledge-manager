from langchain_chroma import Chroma
import chromadb
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
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

    # ❌ DO NOT call vectordb.persist() (removed in new versions)

    print("Embeddings stored successfully.")


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

    top_k = getattr(config, "TOP_K", 3)
    retrieval_mode = getattr(config, "RETRIEVAL_MODE", "vector").lower().strip()

    vector_retriever = vectordb.as_retriever(search_kwargs={"k": top_k})

    if retrieval_mode == "vector":
        print("Retriever mode: vector")
        return vector_retriever

    # Build corpus for lexical retrieval from existing Chroma collection
    records = vectordb.get(include=["documents", "metadatas"])
    docs = records.get("documents", []) or []
    metas = records.get("metadatas", []) or []

    bm25_docs = []
    for idx, text in enumerate(docs):
        if not text:
            continue
        metadata = metas[idx] if idx < len(metas) and metas[idx] else {}
        bm25_docs.append(Document(page_content=text, metadata=metadata))

    # If DB is empty, keep vector retriever behavior
    if not bm25_docs:
        print("Retriever mode: vector (fallback, empty BM25 corpus)")
        return vector_retriever

    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = int(getattr(config, "BM25_K", top_k))

    if retrieval_mode == "bm25":
        print("Retriever mode: bm25")
        return bm25_retriever

    # Default: hybrid (vector + bm25)
    weights = getattr(config, "HYBRID_WEIGHTS", [0.7, 0.3])
    if not isinstance(weights, (list, tuple)) or len(weights) != 2:
        weights = [0.7, 0.3]

    print(f"Retriever mode: hybrid (weights={weights})")
    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=list(weights),
    )


# ======================================================
# DOCUMENT STATISTICS
# ======================================================
def get_document_statistics():
    """
    Retrieves statistics about documents and chunks stored in ChromaDB.
    """
    try:
        client = chromadb.PersistentClient(
            path=config.PERSIST_DIRECTORY
        )

        collection = client.get_or_create_collection(
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
