# Metadata-Augmented Chunking

This repository now uses a metadata-augmented chunking strategy when ingesting PDFs. The goals are:

- Preserve semantic boundaries (headings/sections) when possible.
- Avoid chopping paragraphs in awkward places.
- Attach rich metadata to each chunk to improve retrieval and debugging.

Behavior

- Pages are loaded via `PyPDFLoader`.
- Pages are split into paragraphs by detecting double newlines.
- Short paragraphs that look like headings are marked as `section` metadata.
- Long paragraphs are split using `RecursiveCharacterTextSplitter` to keep chunks near the configured `chunk_size`.
- Each chunk receives metadata: `page`, `section` (optional), `chunk_id`, `start_char`, and `end_char`.

Reducing prompt size / token usage

- The retriever now returns only the top `k` documents (default `k=3`) to avoid creating excessively large prompts that can exceed model token limits.
- Adjust `TOP_K` in `src/config.py` if you need more/fewer documents returned.

Why this helps

- Retrieval can prefer chunks from the same `section` or page.
- Debugging retrieval is easier with start/end offsets and chunk identifiers.

Configuration

- The default `chunk_size` is 1000 characters with 200 overlap. These can be tuned in `src.doc_process.load_and_split_pdf`.
