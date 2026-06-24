"""Extract text from PDFs and split it into overlapping chunks."""
from typing import List, Dict
import fitz  # PyMuPDF


def extract_pdf_pages(file_bytes: bytes) -> List[Dict]:
    """Return a list of {page, text} dicts, one per page."""
    pages = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                pages.append({"page": i + 1, "text": text})
    return pages


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Word-based sliding window chunker."""
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        chunk = words[start : start + chunk_size]
        if chunk:
            chunks.append(" ".join(chunk))
        if start + chunk_size >= len(words):
            break
    return chunks


def process_pdf(
    file_bytes: bytes,
    doc_id: str,
    doc_name: str,
    chunk_size: int,
    overlap: int,
) -> List[Dict]:
    """Return chunk records ready for embedding + upsert."""
    pages = extract_pdf_pages(file_bytes)
    records = []
    idx = 0
    for p in pages:
        for c in chunk_text(p["text"], chunk_size, overlap):
            records.append(
                {
                    "id": f"{doc_id}_{idx}",
                    "text": c,
                    "metadata": {
                        "doc_id": doc_id,
                        "doc_name": doc_name,
                        "page": p["page"],
                        "text": c,
                    },
                }
            )
            idx += 1
    return records
