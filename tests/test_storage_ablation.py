"""
Unit tests for Exact Brute-Force Vector Baseline and HNSW Ablation.
"""
import pytest
import numpy as np

from app.storage.brute_force_index import ExactBruteForceIndex
from app.storage.hnsw_ablation import HNSWAblationEngine, HNSWConfig


def test_exact_brute_force_index():
    """Verify O(N*D) linear scan returns mathematically exact top-K cosine rankings."""
    dim = 4
    bf = ExactBruteForceIndex(dimension=dim)

    # 3 orthogonal / parallel vectors
    v1 = np.array([1.0, 0.0, 0.0, 0.0])  # Identical to q
    v2 = np.array([0.5, 0.5, 0.0, 0.0])  # Cosine ~ 0.707
    v3 = np.array([0.0, 1.0, 0.0, 0.0])  # Orthogonal Cosine == 0.0

    bf.add("doc_exact", v1)
    bf.add("doc_partial", v2)
    bf.add("doc_ortho", v3)

    query = np.array([1.0, 0.0, 0.0, 0.0])
    results = bf.search(query, top_k=3)

    assert len(results) == 3
    assert results[0][0] == "doc_exact"
    assert pytest.approx(results[0][1], rel=1e-3) == 1.0
    assert results[1][0] == "doc_partial"
    assert pytest.approx(results[1][1], rel=1e-3) == 0.7071
    assert results[2][0] == "doc_ortho"
    assert pytest.approx(results[2][1], abs=1e-5) == 0.0


def test_hnsw_ablation_engine_run():
    """Verify HNSW ablation harness constructs graph and evaluates recall against ground truth."""
    engine = HNSWAblationEngine()
    dim = 8
    corpus_ids = [f"doc_{i}" for i in range(16)]
    corpus_vecs = np.random.randn(16, dim).astype(np.float32)
    query_vecs = np.random.randn(4, dim).astype(np.float32)

    # Ground truth
    bf = ExactBruteForceIndex(dimension=dim)
    bf.add_batch(corpus_ids, corpus_vecs)
    ground_truth = bf.compute_ground_truth_sets(query_vecs, top_k=5)

    config = HNSWConfig(m=16, ef_construct=64, ef_search=20)
    metrics = engine.evaluate_configuration(
        config=config,
        corpus_ids=corpus_ids,
        corpus_vectors=corpus_vecs,
        query_vectors=query_vecs,
        ground_truth=ground_truth,
        k=5
    )

    assert metrics.m == 16
    assert metrics.ef_construct == 64
    assert metrics.ef_search == 20
    assert metrics.build_time_sec >= 0.0
    assert metrics.qps > 0.0
    assert 0.0 <= metrics.recall_at_k <= 1.0
