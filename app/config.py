import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection: str = os.getenv("COLLECTION_NAME", "pdf_chunks")
    model_name: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    model_cache_dir: str = os.getenv("MODEL_CACHE_DIR", "/models")
    vector_size: int = int(os.getenv("VECTOR_SIZE", 384))
    top_k: int = int(os.getenv("TOP_K", 5))
    max_query_chars: int = int(os.getenv("MAX_QUERY_CHARS", 1000))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", 800))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 120))
    min_chunk_chars: int = int(os.getenv("MIN_CHUNK_SIZE", 50))


settings = Settings()
