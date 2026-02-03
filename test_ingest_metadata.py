"""Test to verify ingest endpoint stores documents with metadata in ChromaDB."""
import tempfile
import shutil
import os
from langchain.schema import Document
from langchain_chroma import Chroma
from src import embedding_gen, config, vectorstore_manager


def test_ingest_with_metadata_storage():
    """Verify that documents ingested are stored in ChromaDB with all metadata fields."""
    # Create a temp directory for this test's ChromaDB
    test_db_dir = tempfile.mkdtemp()
    test_collection = "test_metadata_collection"
    original_persist_dir = config.PERSIST_DIRECTORY
    original_collection_name = config.COLLECTION_NAME

    try:
        # Create fake documents with metadata-augmented chunks
        docs = [
            Document(
                page_content="INTRODUCTION:\n\nThis is the introduction section.",
                metadata={"page": 0, "section": "INTRODUCTION", "chunk_id": "p0_para0", "start_char": 0, "end_char": 50}
            ),
            Document(
                page_content="This is detailed content in the introduction.",
                metadata={"page": 0, "section": "INTRODUCTION", "chunk_id": "p0_para1", "start_char": 50, "end_char": 100}
            ),
            Document(
                page_content="METHODS:\n\nOur methodology is as follows.",
                metadata={"page": 1, "section": "METHODS", "chunk_id": "p1_para0", "start_char": 0, "end_char": 50}
            ),
        ]

        # Use a real embeddings model or mock
        try:
            embeddings = embedding_gen.get_embeddings()
        except Exception:
            # Simple mock embeddings for testing if real model fails
            class MockEmbeddings:
                def embed_documents(self, texts):
                    return [[0.1] * 384 for _ in texts]

                def embed_query(self, text):
                    return [0.1] * 384

            embeddings = MockEmbeddings()

        # Temporarily change config to use test DB
        config.PERSIST_DIRECTORY = test_db_dir
        config.COLLECTION_NAME = test_collection

        # Store the documents in ChromaDB
        vectorstore_manager.create_and_store_embeddings(docs, embeddings)

        # Retrieve and verify the data was stored with metadata
        vectorstore = Chroma(
            persist_directory=test_db_dir,
            embedding_function=embeddings,
            collection_name=test_collection
        )
        collection = vectorstore._collection
        all_data = collection.get()

        # Assertions
        assert len(all_data["ids"]) == 3, f"Expected 3 documents, got {len(all_data['ids'])}"
        print(f"✅ Correct number of documents stored: {len(all_data['ids'])}")

        # Check metadata presence
        metadatas = all_data["metadatas"]
        assert all("chunk_id" in m for m in metadatas), "Missing chunk_id in metadata"
        assert all("page" in m for m in metadatas), "Missing page in metadata"
        assert all("start_char" in m for m in metadatas), "Missing start_char in metadata"
        assert all("end_char" in m for m in metadatas), "Missing end_char in metadata"
        print("✅ All required metadata fields present: chunk_id, page, start_char, end_char")

        # Check section metadata (optional but enriching)
        section_count = sum(1 for m in metadatas if "section" in m)
        assert section_count >= 2, f"Expected section metadata in at least 2 docs, got {section_count}"
        print(f"✅ Section metadata present in {section_count} documents")

        # Verify specific section values
        sections = {m.get("section") for m in metadatas if "section" in m}
        assert "INTRODUCTION" in sections, "Missing INTRODUCTION section"
        assert "METHODS" in sections, "Missing METHODS section"
        print(f"✅ Sections detected: {sections}")

        # Verify content is stored
        contents = all_data["documents"]
        assert len(contents) == 3, "Content not properly stored"
        assert any("introduction" in c.lower() for c in contents), "Introduction content not found"
        assert any("methodology" in c.lower() for c in contents), "Methods content not found"
        print("✅ Document content properly stored")

        # Test retrieval via retriever (which will use TOP_K)
        retriever = vectorstore_manager.get_retriever(embeddings)
        query = "introduction"
        retrieved = retriever.get_relevant_documents(query)
        assert len(retrieved) > 0, "Retriever returned no documents"
        print(f"✅ Retriever successfully returned {len(retrieved)} documents for query: '{query}'")

        # Verify metadata is preserved in retrieval
        assert all(hasattr(doc, "metadata") for doc in retrieved), "Retrieved docs missing metadata"
        for doc in retrieved:
            assert "chunk_id" in doc.metadata, f"Retrieved doc missing chunk_id: {doc.metadata}"
        print("✅ Retrieved documents contain all metadata")

        print("\n✅✅✅ All metadata storage tests passed!")

    finally:
        # Restore original config
        config.PERSIST_DIRECTORY = original_persist_dir
        config.COLLECTION_NAME = original_collection_name

        # Cleanup
        if os.path.exists(test_db_dir):
            shutil.rmtree(test_db_dir)


if __name__ == "__main__":
    test_ingest_with_metadata_storage()
