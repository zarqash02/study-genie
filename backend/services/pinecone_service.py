"""Pinecone client: index creation, upsert, query, delete."""
from typing import List, Dict, Optional
from pinecone import Pinecone, ServerlessSpec
from backend.config import config

_pc = None
_index = None


def get_client() -> Pinecone:
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=config.PINECONE_API_KEY)
    return _pc


def get_index():
    global _index
    if _index is not None:
        return _index

    pc = get_client()
    existing = [i["name"] for i in pc.list_indexes()]
    if config.PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=config.PINECONE_INDEX_NAME,
            dimension=config.EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION
            ),
        )
    _index = pc.Index(config.PINECONE_INDEX_NAME)
    return _index


def upsert_chunks(records: List[Dict], vectors: List[List[float]], namespace: str = "default"):
    """records: [{id, text, metadata}, ...]  vectors: aligned embeddings."""
    index = get_index()
    payload = [
        {"id": r["id"], "values": v, "metadata": r["metadata"]}
        for r, v in zip(records, vectors)
    ]
    # batch to stay under Pinecone request size limits
    batch = 100
    for i in range(0, len(payload), batch):
        index.upsert(vectors=payload[i : i + batch], namespace=namespace)


def query(
    vector: List[float],
    top_k: int,
    namespace: str = "default",
    doc_ids: Optional[List[str]] = None,
) -> List[Dict]:
    index = get_index()
    flt = {"doc_id": {"$in": doc_ids}} if doc_ids else None
    res = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace,
        filter=flt,
    )
    return [
        {"score": m["score"], "metadata": m["metadata"]}
        for m in res.get("matches", [])
    ]


def delete_doc(doc_id: str, namespace: str = "default"):
    index = get_index()
    index.delete(filter={"doc_id": {"$eq": doc_id}}, namespace=namespace)
