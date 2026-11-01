import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.endpoints import router as api_router
from app.api.ui import ui_router
from app.core.pipeline import HybridRAGEngine
from app.evaluation.dataset import DEFAULT_CORPUS
from app.config import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("hybrid-rag")

BASE_DIR = Path(__file__).resolve().parent.parent

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle startup and shutdown events."""
    logger.info("Initializing Advanced Hybrid RAG Engine in application state...")
    engine = HybridRAGEngine()
    app.state.engine = engine

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
        "(Qdrant + SentenceTransformers), Sparse BM25+ Inverted Index via Cormack Reciprocal Rank Fusion (RRF), "
        "and Cross-Encoder Rerankers with an automated Quantitative Evaluation Benchmark suite."
    ),
    version="1.1.0",
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

# Mount static asset directory for UI
static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Mount API and UI routes
app.include_router(api_router)
app.include_router(ui_router)

@app.get("/health", tags=["Health"])
def health_check(request: Request):
    """Health check endpoint providing engine status."""
    engine = getattr(request.app.state, "engine", None)
    indexed_chunks = engine.vector_store.count() if engine else 0
    return {
        "status": "healthy",
        "service": "advanced-hybrid-rag",
        "version": "1.1.0",
        "indexed_chunks": indexed_chunks
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=True)
