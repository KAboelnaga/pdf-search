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
    data_dir: str = os.getenv("DATA_DIR", "/data")

    def __post_init__(self):
        if self.chunk_size <= 0:
            raise ValueError(f"CHUNK_SIZE ({self.chunk_size}) must be a positive integer.")
        if self.chunk_overlap < 0:
            raise ValueError(f"CHUNK_OVERLAP ({self.chunk_overlap}) must be a non-negative integer.")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"CHUNK_OVERLAP ({self.chunk_overlap}) must be less than CHUNK_SIZE ({self.chunk_size}).")
        if self.top_k <= 0:
            raise ValueError(f"TOP_K ({self.top_k}) must be a positive integer.")
        if self.vector_size <= 0:
            raise ValueError(f"VECTOR_SIZE ({self.vector_size}) must be a positive integer.")
        if self.max_query_chars <= 0:
            raise ValueError(f"MAX_QUERY_CHARS ({self.max_query_chars}) must be a positive integer.")


settings = Settings()
