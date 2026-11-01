import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.models.schemas import RetrievalMode

client = TestClient(app)


def test_health_check():
    """Verify health endpoint returns status 200 and valid json."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "advanced-hybrid-rag"
    assert "version" in data
    assert "indexed_chunks" in data


def test_stats_endpoint():
    """Verify stats endpoint returns active index counters."""
    response = client.get("/api/v1/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_chunks_vector_store" in data
    assert "total_chunks_bm25" in data
    assert "embedding_model" in data
    assert "reranker_model" in data


def test_ingest_empty_validation():
    """Verify 400 validation error when document list is empty."""
    response = client.post("/api/v1/documents", json={"documents": []})
    assert response.status_code == 400
    assert "Document list cannot be empty" in response.json()["detail"]


def test_query_empty_validation():
    """Verify 400 validation error when query is empty."""
    response = client.post("/api/v1/query", json={"query": "   ", "mode": "reranked"})
    assert response.status_code == 400
    assert "Query cannot be empty" in response.json()["detail"]


def test_query_end_to_end():
    """Verify end-to-end query execution across all 3 modes."""
    # Seed a document first
    doc_payload = {
        "documents": [
            {
                "id": "doc_api_test",
                "title": "API Test Specification",
                "text": "The secret verification code for the API gateway test is XYZ-987654.",
                "metadata": {"env": "test"}
            }
        ]
    }
    ingest_resp = client.post("/api/v1/documents", json=doc_payload)
    assert ingest_resp.status_code == 200
    assert ingest_resp.json()["status"] == "success"

    # Query using Hybrid mode
    query_payload = {
        "query": "What is the secret verification code for the API gateway?",
        "mode": "hybrid",
        "top_k": 3,
        "generate_answer": True
    }
    query_resp = client.post("/api/v1/query", json=query_payload)
    assert query_resp.status_code == 200
    res_data = query_resp.json()
    assert res_data["query"] == query_payload["query"]
    assert res_data["mode"] == "hybrid"
    assert len(res_data["retrieved_chunks"]) > 0
    assert "latencies" in res_data
    assert res_data["answer"] is not None


def test_retrieve_endpoint():
    """Verify retrieve-only endpoint returns telemetry without synthesis."""
    req_payload = {
        "query": "verification code",
        "mode": "reranked",
        "top_k": 2
    }
    resp = client.post("/api/v1/retrieve", json=req_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "retrieved_chunks" in data
    assert "latencies" in data
    assert "query" in data
