import types

from src import doc_process


class FakePage:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = metadata or {}


def test_load_and_split_pdf_basic(monkeypatch):
    # Prepare fake pages with headings and long paragraphs
    page1 = FakePage("INTRODUCTION:\n\nThis is a short paragraph.\n\nSECTION TWO:\n\n" +
                     "A " * 200)
    page2 = FakePage("Another page with a single long paragraph " + "B " * 300)

    class FakeLoader:
        def __init__(self, path):
            self.path = path

        def load(self):
            return [page1, page2]

    # Patch the PyPDFLoader used inside src.doc_process
    monkeypatch.setattr(doc_process, "PyPDFLoader", FakeLoader)

    chunks = doc_process.load_and_split_pdf("dummy.pdf", chunk_size=100, chunk_overlap=10)

    # Expect at least one chunk per page and metadata present
    assert len(chunks) >= 2

    for c in chunks:
        assert hasattr(c, "page_content")
        assert hasattr(c, "metadata")
        assert "chunk_id" in c.metadata
        assert "page" in c.metadata
        assert "start_char" in c.metadata and "end_char" in c.metadata
