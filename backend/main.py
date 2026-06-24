"""Study Genie — FastAPI backend.

Endpoints:
  GET  /health
  POST /upload     multipart PDF -> chunk, embed, upsert to Pinecone
  POST /ask        JSON {query, doc_ids?} -> grounded answer + sources
  POST /summarize  JSON {doc_ids?} -> structured summary
  POST /quiz       JSON {doc_ids?, n?} -> MCQs
  POST /flashcards JSON {doc_ids?, n?} -> Q/A pairs
  DELETE /document/{doc_id}
"""
import uuid
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from backend.config import config
from backend.services import pdf_service, embedding_service, pinecone_service, llm_service

app = FastAPI(title="Study Genie API", version="1.0.0")


class AskRequest(BaseModel):
    query: str
    doc_ids: Optional[List[str]] = None
    top_k: Optional[int] = None


class DocIdsRequest(BaseModel):
    doc_ids: Optional[List[str]] = None


class GenRequest(BaseModel):
    doc_ids: Optional[List[str]] = None
    n: Optional[int] = 5


@app.on_event("startup")
def _startup():
    # Fail fast if secrets missing; warm the embedding model.
    config.validate()
    embedding_service.get_model()
    pinecone_service.get_index()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        file_bytes = await file.read()
        doc_id = str(uuid.uuid4())[:8]

        records = pdf_service.process_pdf(
            file_bytes=file_bytes,
            doc_id=doc_id,
            doc_name=file.filename,
            chunk_size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP,
        )
        if not records:
            raise HTTPException(status_code=400, detail="No extractable text found in PDF.")

        vectors = embedding_service.embed_texts([r["text"] for r in records])
        pinecone_service.upsert_chunks(records, vectors)

        return {
            "doc_id": doc_id,
            "doc_name": file.filename,
            "chunks": len(records),
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[UPLOAD ERROR] {e}\n{tb}", flush=True)
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


@app.post("/ask")
def ask(req: AskRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query.")

    vec = embedding_service.embed_query(req.query)
    top_k = req.top_k or config.TOP_K
    matches = pinecone_service.query(vec, top_k=top_k, doc_ids=req.doc_ids)

    if not matches:
        return {
            "answer": "I don't know based on the provided material.",
            "sources": [],
        }

    answer = llm_service.answer_question(req.query, matches)
    sources = [
        {
            "doc_name": m["metadata"].get("doc_name"),
            "page": m["metadata"].get("page"),
            "score": round(m["score"], 3),
            "snippet": (m["metadata"].get("text", "")[:220] + "…"),
        }
        for m in matches
    ]
    return {"answer": answer, "sources": sources}


def _fetch_context(doc_ids: Optional[List[str]], k: int = 8):
    """Pull representative chunks for summary/quiz/flashcards.
    Uses a generic query vector to retrieve; if doc_ids given, filtered to those docs.
    """
    seed = embedding_service.embed_query("key concepts and main ideas overview")
    return pinecone_service.query(seed, top_k=k, doc_ids=doc_ids)


@app.post("/summarize")
def summarize(req: DocIdsRequest):
    chunks = _fetch_context(req.doc_ids, k=10)
    if not chunks:
        raise HTTPException(status_code=404, detail="No content found.")
    return {"summary": llm_service.summarize(chunks)}


@app.post("/quiz")
def quiz(req: GenRequest):
    chunks = _fetch_context(req.doc_ids, k=10)
    if not chunks:
        raise HTTPException(status_code=404, detail="No content found.")
    return {"quiz": llm_service.make_quiz(chunks, n=req.n or 5)}


@app.post("/flashcards")
def flashcards(req: GenRequest):
    chunks = _fetch_context(req.doc_ids, k=10)
    if not chunks:
        raise HTTPException(status_code=404, detail="No content found.")
    return {"flashcards": llm_service.make_flashcards(chunks, n=req.n or 8)}


@app.delete("/document/{doc_id}")
def delete_document(doc_id: str):
    pinecone_service.delete_doc(doc_id)
    return {"deleted": doc_id}
