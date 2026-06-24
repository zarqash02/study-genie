"""Local embeddings via sentence-transformers. Model loads once."""
from typing import List
from sentence_transformers import SentenceTransformer
from backend.config import config

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_model()
    vecs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return vecs.tolist()


def embed_query(query: str) -> List[float]:
    return embed_texts([query])[0]
