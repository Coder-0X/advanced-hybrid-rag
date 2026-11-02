"""
Unit tests for Hierarchical Cache, Anti-Stampede SingleFlight, and Threshold Calibrator.
"""
import asyncio
import time
import pytest
import numpy as np

from app.cache.single_flight import SingleFlight
from app.cache.hierarchical_cache import HierarchicalCache
from app.cache.threshold_evaluator import SemanticCacheThresholdCalibrator


@pytest.mark.anyio
async def test_single_flight_coalescing():
    """Verify concurrent requests for same key are coalesced to single execution."""
    sf = SingleFlight()
    executions = 0

    async def worker():
        nonlocal executions
        executions += 1
        await asyncio.sleep(0.01)
        return "computed_result"

    # Launch 10 simultaneous coroutines
    tasks = [sf.execute("identical_key", worker) for _ in range(10)]
    results = await asyncio.gather(*tasks)

    assert len(results) == 10
    assert all(r == "computed_result" for r in results)
    assert executions == 1  # Exactly 1 downstream execution
    assert sf.stats()["coalesced_calls"] == 9


def test_hierarchical_cache_l1_exact_and_ast_normalization():
    """Verify L1 exact matching normalizes whitespace and casing."""
    cache = HierarchicalCache(l1_capacity=10, l1_ttl_sec=60.0)

    cache.put(
        query="What is   Cormack RRF? ",
        query_embedding=None,
        response={"answer": "Canonical k = 60"}
    )

    # Probe with different whitespace and casing
    hit = cache.get(query="what is cormack rrf?")
    assert hit is not None
    tag, val = hit
    assert tag == "L1_EXACT"
    assert val["answer"] == "Canonical k = 60"


def test_hierarchical_cache_l2_semantic_vector():
    """Verify L2 semantic cache returns hit when vector cosine similarity >= threshold."""
    cache = HierarchicalCache(l1_capacity=10, l2_threshold=0.90)

    # Base vector [1, 0, 0, 0]
    base_v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    cache.put(
        query="query alpha",
        query_embedding=base_v,
        response={"data": "alpha payload"}
    )

    # Near vector [0.98, 0.1, 0.0, 0.0] -> cosine ~ 0.99
    near_v = np.array([0.98, 0.1, 0.0, 0.0], dtype=np.float32)
    hit = cache.get(query="query beta different wording", query_embedding=near_v)

    assert hit is not None
    tag, val = hit
    assert tag == "L2_SEMANTIC"
    assert val["data"] == "alpha payload"

    # Distant vector [0.0, 1.0, 0.0, 0.0] -> cosine == 0.0 -> Miss
    distant_v = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    miss = cache.get(query="query gamma", query_embedding=distant_v)
    assert miss is None


def test_threshold_calibrator_youden_j():
    """Verify threshold calibrator finds threshold maximizing TPR - FPR."""
    sims = np.array([0.98, 0.95, 0.92, 0.85, 0.70, 0.60])
    labels = np.array([1, 1, 1, 0, 0, 0])

    best_tau, best_pt, roc_pts = SemanticCacheThresholdCalibrator.find_optimal_threshold(
        cosine_similarities=sims,
        ground_truth_labels=labels
    )

    # At threshold 0.90: all 3 positives >= 0.90 (TPR = 1.0), all 3 negatives < 0.90 (FPR = 0.0) -> Youden's J = 1.0
    assert 0.86 <= best_tau <= 0.92
    assert best_pt.youden_j == 1.0
    assert best_pt.fpr == 0.0
    assert best_pt.tpr == 1.0
