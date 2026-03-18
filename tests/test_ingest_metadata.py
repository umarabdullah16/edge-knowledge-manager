"""
Test to verify ingest endpoint stores documents with metadata in ChromaDB.
"""
import tempfile
import shutil
import os
import time
import gc

try:
    from langchain_core.documents import Document
except ImportError:  # Backward compatibility with older LangChain
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
        # --------------------------------------------------
        # Create fake documents with rich metadata
        # --------------------------------------------------
        docs = [
            Document(
                page_content="INTRODUCTION:\n\nThis is the introduction section.",
                metadata={
                    "page": 0,
                    "section": "INTRODUCTION",
                    "chunk_id": "p0_para0",
                    "start_char": 0,
                    "end_char": 50,
                },
            ),
            Document(
                page_content="This is detailed content in the introduction.",
                metadata={
                    "page": 0,
                    "section": "INTRODUCTION",
                    "chunk_id": "p0_para1",
                    "start_char": 50,
                    "end_char": 100,
                },
            ),
            Document(
                page_content="METHODS:\n\nOur methodology is as follows.",
                metadata={
                    "page": 1,
                    "section": "METHODS",
                    "chunk_id": "p1_para0",
                    "start_char": 0,
                    "end_char": 50,
                },
            ),
        ]

        # --------------------------------------------------
        # Load embeddings (real or fallback mock)
        # --------------------------------------------------
        try:
            embeddings = embedding_gen.get_embeddings()
        except Exception:
            class MockEmbeddings:
                def embed_documents(self, texts):
                    return [[0.1] * 384 for _ in texts]

                def embed_query(self, text):
                    return [0.1] * 384

            embeddings = MockEmbeddings()

        # --------------------------------------------------
        # Override config for isolated test DB
        # --------------------------------------------------
        config.PERSIST_DIRECTORY = test_db_dir
        config.COLLECTION_NAME = test_collection

        # --------------------------------------------------
        # Ingest documents
        # --------------------------------------------------
        vectorstore_manager.create_and_store_embeddings(docs, embeddings)

        # --------------------------------------------------
        # Verify ChromaDB contents
        # --------------------------------------------------
        vectorstore = Chroma(
            persist_directory=test_db_dir,
            embedding_function=embeddings,
            collection_name=test_collection,
        )

        collection = vectorstore._collection
        all_data = collection.get()

        # ---------------- Assertions ----------------
        assert len(all_data["ids"]) == 3, f"Expected 3 documents, got {len(all_data['ids'])}"
        print(f"✅ Correct number of documents stored: {len(all_data['ids'])}")

        # Metadata checks
        metadatas = all_data["metadatas"]
        assert all("chunk_id" in m for m in metadatas), "Missing chunk_id in metadata"
        assert all("page" in m for m in metadatas), "Missing page in metadata"
        assert all("start_char" in m for m in metadatas), "Missing start_char in metadata"
        assert all("end_char" in m for m in metadatas), "Missing end_char in metadata"
        print("✅ All required metadata fields present")

        # Section metadata
        section_count = sum(1 for m in metadatas if "section" in m)
        assert section_count >= 2, f"Expected section metadata in at least 2 docs, got {section_count}"
        print(f"✅ Section metadata present in {section_count} documents")

        sections = {m.get("section") for m in metadatas if "section" in m}
        assert "INTRODUCTION" in sections
        assert "METHODS" in sections
        print(f"✅ Sections detected: {sections}")

        # Content verification
        contents = all_data["documents"]
        assert len(contents) == 3
        assert any("introduction" in c.lower() for c in contents)
        assert any("methodology" in c.lower() for c in contents)
        print("✅ Document content properly stored")

        # --------------------------------------------------
        # Retrieval test
        # --------------------------------------------------
        retriever = vectorstore_manager.get_retriever(embeddings)
        retrieved = retriever.get_relevant_documents("introduction")
        assert len(retrieved) > 0
        print(f"✅ Retriever returned {len(retrieved)} documents")

        for doc in retrieved:
            assert "chunk_id" in doc.metadata
        print("✅ Retrieved documents preserve metadata")

        print("\n✅✅✅ All metadata storage tests passed!")

    finally:
        # --------------------------------------------------
        # Restore config
        # --------------------------------------------------
        config.PERSIST_DIRECTORY = original_persist_dir
        config.COLLECTION_NAME = original_collection_name

        # --------------------------------------------------
        # Windows-safe cleanup (VERY IMPORTANT)
        # --------------------------------------------------
        try:
            del vectorstore
        except Exception:
            pass

        gc.collect()
        time.sleep(0.5)

        if os.path.exists(test_db_dir):
            try:
                shutil.rmtree(test_db_dir)
            except PermissionError:
                print("⚠️ ChromaDB files still locked on Windows; skipping cleanup.")


if __name__ == "__main__":
    test_ingest_with_metadata_storage()
