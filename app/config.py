import os
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict

class Settings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Base paths
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    STORAGE_PATH: Path = DATA_DIR / "qdrant_db"
    BENCHMARK_RESULTS_DIR: Path = DATA_DIR / "benchmarks"
    SAMPLE_DATA_DIR: Path = PROJECT_ROOT / "sample_data"

    # Retrieval Models
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # Vector DB
    QDRANT_COLLECTION: str = "rag_documents"
    USE_IN_MEMORY_DB: bool = os.getenv("USE_IN_MEMORY_DB", "true").lower() in ("true", "1", "yes")

    # Retrieval defaults
    DENSE_TOP_K: int = 10
    SPARSE_TOP_K: int = 10
    RRF_K: int = 60
    DENSE_WEIGHT: float = 0.5
    SPARSE_WEIGHT: float = 0.5
    RERANK_TOP_K: int = 5
    FINAL_TOP_K: int = 5

    # Document Chunking
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # API and Generation
    API_PORT: int = int(os.getenv("PORT", "8000"))
    API_HOST: str = os.getenv("HOST", "0.0.0.0")

settings = Settings()

# Ensure directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.BENCHMARK_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
