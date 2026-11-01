import pytest
from app.core.reranker import CrossEncoderReranker
from app.models.schemas import RetrievedChunk

def test_reranker_reordering():
    reranker = CrossEncoderReranker()

    chunk_irrelevant = RetrievedChunk(
        chunk_id="c_irr",
        document_id="d1",
        text="Chocolate cake recipe with strawberries and vanilla cream.",
        final_score=0.9
    )
    chunk_relevant = RetrievedChunk(
        chunk_id="c_rel",
        document_id="d2",
        text="Kubernetes Horizontal Pod Autoscaler adjusts replica count based on CPU target metrics.",
        final_score=0.5
    )

    # Initial order: irrelevant was falsely ranked higher (e.g. by vector hallucination)
    candidates = [chunk_irrelevant, chunk_relevant]
    query = "How does Horizontal Pod Autoscaler work in Kubernetes?"

    reranked = reranker.rerank(query, candidates, top_k=2)

    # Cross-encoder should promote the relevant chunk to #1
    assert len(reranked) == 2
    assert reranked[0].chunk_id == "c_rel"
    assert reranked[0].final_score > reranked[1].final_score
    assert reranked[0].rerank_score is not None

def test_sigmoid_monotonic():
    assert CrossEncoderReranker._sigmoid(0.0) == 0.5
    assert CrossEncoderReranker._sigmoid(10.0) > 0.99
    assert CrossEncoderReranker._sigmoid(-10.0) < 0.01
