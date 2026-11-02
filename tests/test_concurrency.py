"""
Unit tests for Dynamic Micro-Batching Concurrency Engine.
"""
import asyncio
import pytest
from typing import List, Tuple
from app.concurrency.batch_queue import DynamicMicroBatcher


def mock_inference_fn(pairs: List[Tuple[str, str]]) -> List[float]:
    """Mock tensor inference function returning deterministic scores."""
    return [len(p[0]) / (len(p[1]) + 1.0) for p in pairs]


@pytest.mark.anyio
async def test_dynamic_micro_batcher_lifecycle():
    """Verify batcher starts, executes batched requests, and stops cleanly."""
    batcher = DynamicMicroBatcher(
        inference_fn=mock_inference_fn,
        max_batch_size=8,
        window_ms=10.0,
        worker_threads=2
    )
    await batcher.start()

    score = await batcher.predict("short query", "long passage text for testing")
    assert isinstance(score, float)
    assert score > 0.0

    stats = batcher.stats()
    assert stats["total_batches_executed"] >= 1
    assert stats["total_queries_processed"] >= 1

    await batcher.stop()


@pytest.mark.anyio
async def test_concurrent_sliding_window_accumulation():
    """Verify multiple concurrent tasks are accumulated into a single batch."""
    batcher = DynamicMicroBatcher(
        inference_fn=mock_inference_fn,
        max_batch_size=16,
        window_ms=25.0,
        worker_threads=2
    )
    await batcher.start()

    queries = [f"query_{i}" for i in range(10)]
    passages = [f"passage_{i}" for i in range(10)]

    # Fire 10 simultaneous coroutines
    tasks = [batcher.predict(q, p) for q, p in zip(queries, passages)]
    results = await asyncio.gather(*tasks)

    assert len(results) == 10
    stats = batcher.stats()
    # High batch utilization: 10 queries should be batched together
    assert stats["total_batches_executed"] <= 2
    assert stats["avg_batch_utilization"] >= 5.0

    await batcher.stop()
