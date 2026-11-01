import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import router as api_router, engine
from app.api.ui import ui_router
from app.evaluation.dataset import DEFAULT_CORPUS
from app.config import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("hybrid-rag")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle startup and shutdown events."""
    logger.info("Initializing Advanced Hybrid RAG Engine...")
    # Seed default sample corpus if indices are empty
    if engine.vector_store.count() == 0:
        logger.info("Auto-indexing sample technical knowledge base on startup...")
        engine.ingest_documents(DEFAULT_CORPUS)
        logger.info(f"Initialized with {engine.vector_store.count()} indexed chunks.")
    yield
    logger.info("Shutting down Hybrid RAG Engine.")

app = FastAPI(
    title="Advanced Hybrid RAG Engine",
    description=(
        "Production-grade Hybrid Information Retrieval engine combining Dense Embeddings "
        "(Qdrant + SentenceTransformers), Sparse Okapi BM25 via Cormack Reciprocal Rank Fusion (RRF), "
        "and Cross-Encoder Rerankers with an automated Quantitative Evaluation Benchmark suite."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(api_router)
app.include_router(ui_router)

@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "advanced-hybrid-rag",
        "version": "1.0.0",
        "indexed_chunks": engine.vector_store.count()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=True)
