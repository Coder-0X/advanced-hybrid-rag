import pytest
from app.core.bm25_retriever import BM25Retriever
from app.models.schemas import DocumentChunk

def test_bm25_tokenize():
    tokens = BM25Retriever.tokenize("The Quick Brown FOX jumps over 42 lazy dogs!")
    assert "quick" in tokens
    assert "brown" in tokens
    assert "fox" in tokens
    assert "the" not in tokens  # Stopword removed
    assert "42" in tokens

def test_bm25_retrieval_ranking():
    retriever = BM25Retriever()
    chunk1 = DocumentChunk(
        id="c1",
        document_id="d1",
        chunk_index=0,
        text="Kubernetes Horizontal Pod Autoscaler scales pods automatically based on CPU utilization."
    )
    chunk2 = DocumentChunk(
        id="c2",
        document_id="d2",
        chunk_index=0,
        text="PostgreSQL relational database storage and B-Tree indexing mechanisms."
    )
    retriever.add_chunks([chunk1, chunk2])

    results = retriever.retrieve("Kubernetes Pod Autoscaler", top_k=2)
    assert len(results) == 2
    top_chunk, score, rank = results[0]
    assert top_chunk.id == "c1"
    assert rank == 1
    assert score > 0.0

def test_bm25_clear():
    retriever = BM25Retriever()
    chunk = DocumentChunk(id="c1", document_id="d1", chunk_index=0, text="Sample text")
    retriever.add_chunks([chunk])
    assert retriever.total_documents == 1
    retriever.clear()
    assert retriever.total_documents == 0
