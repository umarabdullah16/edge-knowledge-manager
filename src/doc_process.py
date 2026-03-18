
"""
This module is responsible for processing documents, including loading and splitting them into chunks.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
try:
    from langchain_core.documents import Document
except ImportError:  # Backward compatibility with older LangChain
    from langchain.schema import Document
import re


def _is_heading(text: str) -> bool:
    """Heuristic to determine whether a short text chunk is a heading/section title."""
    s = text.strip()
    if not s:
        return False
    # Short, ends with colon, or is mostly title-cased / all-caps
    if len(s) < 120 and (s.endswith(":") or s.isupper()):
        return True
    # Looks like a title (Capitalized words, not a full sentence)
    if len(s.split()) <= 8 and re.match(r"^[A-Z0-9][A-Za-z0-9\-\s,:()]+$", s):
        # avoid matching full sentences (has trailing period)
        return not s.endswith(".")
    return False


def load_and_split_pdf(file_path, chunk_size=1000, chunk_overlap=200):
    """
    Loads a PDF and performs metadata-augmented chunking.

    Behavior:
    - Loads pages via `PyPDFLoader` (each returned Document is usually a page).
    - Splits pages into paragraphs/sections first (preserving headings when detected).
    - For long paragraphs, falls back to a RecursiveCharacterTextSplitter so chunks are near `chunk_size`.
    - Attaches metadata: original page metadata, `page`, `section` (if detected), `chunk_id`, `start_char`, `end_char`.

    Returns a list of LangChain `Document` objects suitable for embedding/storage.
    """
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunks = []
    for page_idx, page in enumerate(pages):
        text = page.page_content
        page_meta = dict(page.metadata) if getattr(page, "metadata", None) else {}

        # split into paragraphs by two or more newlines, fallback to single newlines
        paragraphs = re.split(r"\n{2,}", text)

        char_cursor = 0
        current_section = None
        for para_idx, para in enumerate(paragraphs):
            para = para.strip()
            if not para:
                char_cursor += len(para) + 2
                continue

            # detect if paragraph is a heading/section marker
            if _is_heading(para):
                current_section = para.rstrip(":")

            # If paragraph is small enough, keep as single chunk
            if len(para) <= chunk_size:
                meta = {**page_meta}
                if "page" not in meta:
                    meta["page"] = page_meta.get("page", page_idx)
                if current_section:
                    meta["section"] = current_section
                start = text.find(para, char_cursor)
                if start == -1:
                    start = char_cursor
                end = start + len(para)
                meta.update({"chunk_id": f"p{page_idx}_para{para_idx}", "start_char": start, "end_char": end})
                chunks.append(Document(page_content=para, metadata=meta))
                char_cursor = end
            else:
                # long paragraph: use RecursiveCharacterTextSplitter to create semantically aware chunks
                sub_docs = splitter.split_text(para)
                sub_cursor = 0
                for sub_idx, sub in enumerate(sub_docs):
                    meta = {**page_meta}
                    if "page" not in meta:
                        meta["page"] = page_meta.get("page", page_idx)
                    if current_section:
                        meta["section"] = current_section
                    # approximate start/end within the page text
                    start = text.find(sub, char_cursor + sub_cursor)
                    if start == -1:
                        # fallback: incremental cursor
                        start = char_cursor + sub_cursor
                    end = start + len(sub)
                    meta.update({"chunk_id": f"p{page_idx}_para{para_idx}_s{sub_idx}", "start_char": start, "end_char": end})
                    chunks.append(Document(page_content=sub, metadata=meta))
                    sub_cursor = end - char_cursor
                char_cursor += len(para) + 2

    return chunks
