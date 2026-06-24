"""Central configuration loaded from environment variables."""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Pinecone
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "study-genie")
    PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
    PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

    # Groq (primary LLM provider)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # HuggingFace Inference Providers (fallback LLM)
    HF_TOKEN = os.getenv("HF_TOKEN", "")
    HF_GENERATION_MODEL = os.getenv(
        "HF_GENERATION_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1"
    )

    # Embeddings (runs locally via sentence-transformers)
    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

    # RAG params
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
    TOP_K = int(os.getenv("TOP_K", "4"))

    @classmethod
    def validate(cls):
        """Fail fast on required env vars. Groq is optional (HF is fallback)."""
        missing = []
        if not cls.PINECONE_API_KEY:
            missing.append("PINECONE_API_KEY")
        if not cls.HF_TOKEN and not cls.GROQ_API_KEY:
            missing.append("HF_TOKEN or GROQ_API_KEY (at least one required)")
        if missing:
            raise RuntimeError(
                f"Missing required environment variables: {', '.join(missing)}"
            )


config = Config()