"""
REST API Endpoints for Advanced Hybrid RAG Engine.
Uses FastAPI Dependency Injection for thread-safe, multi-worker engine management.
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from app.models.schemas import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
    LatencyBreakdown,
    BenchmarkReport
)
from app.core.pipeline import HybridRAGEngine
from app.api.dependencies import get_engine
from app.evaluation.benchmark_runner import AutomatedBenchmarkRunner
from app.evaluation.dataset import DEFAULT_CORPUS, DEFAULT_BENCHMARK_DATASET
from app.config import settings

router = APIRouter(prefix="/api/v1", tags=["RAG & Benchmarking"])

@router.post("/documents", response_model=IngestResponse, summary="Ingest and index documents")
def ingest_documents(
    req: IngestRequest,
    engine: HybridRAGEngine = Depends(get_engine)
):
    """Chunk and index documents into both Qdrant vector store and BM25 inverted index."""
    if not req.documents:
        raise HTTPException(status_code=400, detail="Document list cannot be empty.")
    return engine.ingest_documents(req.documents)

@router.post("/query", response_model=QueryResponse, summary="Execute RAG query")
def query_rag(
    req: QueryRequest,
    engine: HybridRAGEngine = Depends(get_engine)
):
    """
    Run end-to-end RAG query using the selected mode:
    - naive: Dense vector search only
    - hybrid: Dense + BM25 merged via Reciprocal Rank Fusion
    - reranked: Hybrid candidate pool refined by Cross-Encoder
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    return engine.query(req)

@router.post("/retrieve", summary="Information Retrieval inspection only")
def retrieve_chunks(
    req: QueryRequest,
    engine: HybridRAGEngine = Depends(get_engine)
):
    """Retrieve chunks and telemetry without generating an answer."""
    chunks, latencies = engine.retrieve(
        query=req.query,
        mode=req.mode,
        top_k=req.top_k,
        dense_k=req.dense_k,
        sparse_k=req.sparse_k,
        dense_weight=req.dense_weight,
        sparse_weight=req.sparse_weight,
        rrf_k=req.rrf_k
    )
    return {
        "query": req.query,
        "mode": req.mode,
        "retrieved_chunks": [c.model_dump() for c in chunks],
        "latencies": latencies.model_dump()
    }

@router.post("/benchmark", response_model=BenchmarkReport, summary="Trigger automated quantitative benchmark")
def run_benchmark(
    engine: HybridRAGEngine = Depends(get_engine)
):
    """
    Runs automated evaluation benchmark comparing Naive vs Hybrid vs Reranked
    across faithfulness, context precision, context recall, hit rate, MRR, and latency.
    """
    runner = AutomatedBenchmarkRunner(engine=engine)
    return runner.run_benchmark()

@router.get("/benchmark/latest", summary="Retrieve most recent benchmark report")
def get_latest_benchmark(
    engine: HybridRAGEngine = Depends(get_engine)
):
    """Returns the latest benchmark report or raises 404 if not run yet."""
    latest_file = settings.BENCHMARK_RESULTS_DIR / "latest_benchmark.json"
    if not latest_file.exists():
        runner = AutomatedBenchmarkRunner(engine=engine)
        return runner.run_benchmark()
    with open(latest_file, "r", encoding="utf-8") as f:
        return json.load(f)

@router.post("/corpus/load-sample", summary="Load sample infrastructure corpus")
def load_sample_corpus(
    engine: HybridRAGEngine = Depends(get_engine)
):
    """Convenience endpoint to seed the vector store and BM25 index with sample documents."""
    resp = engine.ingest_documents(DEFAULT_CORPUS)
    return {
        "status": "success",
        "message": f"Successfully indexed {len(DEFAULT_CORPUS)} documents into Dense and BM25 stores.",
        "details": resp.model_dump()
    }

@router.get("/stats", summary="Index and model diagnostics")
def get_stats(
    engine: HybridRAGEngine = Depends(get_engine)
):
    """Get active index statistics and model configs."""
    return engine.stats()

@router.post("/clear", summary="Clear indices")
def clear_indices(
    engine: HybridRAGEngine = Depends(get_engine)
):
    """Clear all vector and BM25 indices."""
    engine.clear()
    return {"status": "success", "message": "All document indices cleared."}

# Lazy module-level backward compatibility for tests importing `engine` directly
_default_engine = None
def __getattr__(name: str):
    global _default_engine
    if name == "engine":
        if _default_engine is None:
            _default_engine = HybridRAGEngine()
        return _default_engine
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
