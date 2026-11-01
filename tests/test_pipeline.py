import pytest
from app.core.pipeline import HybridRAGEngine
from app.models.schemas import DocumentInput, RetrievalMode, QueryRequest

@pytest.fixture
def test_engine():
    engine = HybridRAGEngine()
    engine.clear()
    docs = [
        DocumentInput(
            id="doc_k8s",
            title="Kubernetes HPA",
            text="Kubernetes Horizontal Pod Autoscaler scales pod replicas using CPU and memory utilization metrics.",
            metadata={"source": "k8s_docs"}
        ),
        DocumentInput(
            id="doc_qdrant",
            title="Qdrant Vectors",
            text="Qdrant performs vector similarity search using Hierarchical Navigable Small World HNSW graphs.",
            metadata={"source": "qdrant_docs"}
        )
    ]
    engine.ingest_documents(docs)
    return engine

def test_pipeline_naive_retrieval(test_engine):
    req = QueryRequest(query="Kubernetes HPA autoscaling", mode=RetrievalMode.NAIVE, top_k=2)
    resp = test_engine.query(req)
    assert len(resp.retrieved_chunks) >= 1
    assert resp.mode == RetrievalMode.NAIVE
    assert resp.latencies.dense_retrieval_ms >= 0
    assert resp.answer is not None
    assert "Kubernetes" in resp.answer

def test_pipeline_hybrid_retrieval(test_engine):
    req = QueryRequest(query="HNSW graph similarity in Qdrant", mode=RetrievalMode.HYBRID, top_k=2)
    resp = test_engine.query(req)
    assert len(resp.retrieved_chunks) >= 1
    assert resp.mode == RetrievalMode.HYBRID
    assert resp.retrieved_chunks[0].rrf_score is not None
    assert resp.latencies.fusion_ms >= 0

def test_pipeline_reranked_retrieval(test_engine):
    req = QueryRequest(query="Qdrant HNSW graphs", mode=RetrievalMode.RERANKED, top_k=2)
    resp = test_engine.query(req)
    assert len(resp.retrieved_chunks) >= 1
    assert resp.mode == RetrievalMode.RERANKED
    assert resp.retrieved_chunks[0].rerank_score is not None
    assert resp.latencies.reranking_ms >= 0
