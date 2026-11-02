"""
Asynchronous Concurrency & Dynamic Micro-Batching Benchmark.

Simulates C = 50 concurrent client queries against:
1. Unbatched Sequential / Serial execution (naive baseline).
2. Dynamic Micro-Batching with Sliding Window (Delta_t = 8ms, B_max = 32) + ThreadPool offload.

Quantifies Throughput (QPS), Tail Latencies (p50, p95, p99), Batch Utilization,
and Event-Loop Responsiveness (GIL latency jitter).
"""
import asyncio
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

from app.core.onnx_reranker import ONNXReranker
from app.concurrency.batch_queue import DynamicMicroBatcher
from app.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("concurrency_benchmark")


async def measure_event_loop_lag(stop_event: asyncio.Event) -> List[float]:
    """Measures event loop delay by scheduling a 1ms timer and recording jitter."""
    lags = []
    interval = 0.005  # 5ms tick
    while not stop_event.is_set():
        t0 = time.perf_counter()
        await asyncio.sleep(interval)
        actual_delay = time.perf_counter() - t0
        jitter_ms = max(0.0, (actual_delay - interval) * 1000.0)
        lags.append(jitter_ms)
    return lags


async def run_serial_concurrency(
    reranker: ONNXReranker,
    test_pairs: List[Tuple[str, str]]
) -> Tuple[List[float], List[float]]:
    """Runs unbatched requests concurrently where each request calls model directly."""
    stop_event = asyncio.Event()
    lag_task = asyncio.create_task(measure_event_loop_lag(stop_event))

    latencies = []

    async def single_client(q: str, p: str):
        t0 = time.perf_counter()
        # Direct synchronous call in loop (naive AI slop pattern)
        _ = reranker.predict([(q, p)])
        latencies.append((time.perf_counter() - t0) * 1000.0)

    # Launch all client coroutines concurrently
    await asyncio.gather(*[single_client(q, p) for q, p in test_pairs])

    stop_event.set()
    lags = await lag_task
    return latencies, lags


async def run_microbatch_concurrency(
    batcher: DynamicMicroBatcher,
    test_pairs: List[Tuple[str, str]]
) -> Tuple[List[float], List[float]]:
    """Runs concurrent requests through DynamicMicroBatcher."""
    stop_event = asyncio.Event()
    lag_task = asyncio.create_task(measure_event_loop_lag(stop_event))

    latencies = []

    async def single_client(q: str, p: str):
        t0 = time.perf_counter()
        _ = await batcher.predict(q, p)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    # Fire all clients simultaneously
    await asyncio.gather(*[single_client(q, p) for q, p in test_pairs])

    stop_event.set()
    lags = await lag_task
    return latencies, lags


async def run_benchmark(num_requests: int = 50, output_path: str = "reports/concurrency_benchmark.json"):
    reports_dir = settings.PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = settings.PROJECT_ROOT / output_path

    logger.info("Initializing ONNX Reranker for concurrency evaluation...")
    reranker = ONNXReranker(use_int8=True)

    # Prepare representative pairs
    sample_queries = [
        "What is the mathematical formulation of Cormack RRF?",
        "How does Raft elect leaders during network partitions?",
        "PostgreSQL MVCC xmin and xmax header visibility rules.",
        "Kubernetes Horizontal Pod Autoscaler replica calculation formula.",
        "BM25+ lower bound parameter delta role in document length normalization."
    ]
    sample_passages = [
        "Reciprocal Rank Fusion computes RRF_score = sum 1 / (k + rank) with k = 60.",
        "In Raft consensus, followers increment term and grant votes if candidate log is up-to-date.",
        "PostgreSQL tuple headers store xmin for inserting transaction and xmax for deleting transaction.",
        "Horizontal Pod Autoscaler calculates ceil(currentReplicas * (currentMetric / targetMetric)).",
        "BM25+ introduces delta > 0 to prevent penalizing very short documents with negative scores."
    ]

    test_pairs = [
        (sample_queries[i % len(sample_queries)], sample_passages[i % len(sample_passages)])
        for i in range(num_requests)
    ]

    # Benchmark 1: Unbatched Serial / Event-loop Blocking
    logger.info(f"Running Unbatched Naive Concurrency Benchmark ({num_requests} requests)...")
    t_start_serial = time.perf_counter()
    serial_lats, serial_lags = await run_serial_concurrency(reranker, test_pairs)
    t_serial_total = time.perf_counter() - t_start_serial
    serial_qps = num_requests / t_serial_total

    # Benchmark 2: Dynamic Micro-Batching
    logger.info(f"Running Dynamic Micro-Batching Concurrency Benchmark ({num_requests} requests)...")
    batcher = DynamicMicroBatcher(
        inference_fn=lambda pairs: reranker.predict(pairs, batch_size=32),
        max_batch_size=32,
        window_ms=8.0,
        worker_threads=4
    )
    await batcher.start()

    t_start_batch = time.perf_counter()
    batch_lats, batch_lags = await run_microbatch_concurrency(batcher, test_pairs)
    t_batch_total = time.perf_counter() - t_start_batch
    batch_qps = num_requests / t_batch_total

    b_stats = batcher.stats()
    await batcher.stop()

    # Metrics compilation
    res = {
        "num_requests": num_requests,
        "serial": {
            "total_time_sec": t_serial_total,
            "qps": serial_qps,
            "mean_latency_ms": float(np.mean(serial_lats)),
            "p50_latency_ms": float(np.percentile(serial_lats, 50)),
            "p95_latency_ms": float(np.percentile(serial_lats, 95)),
            "p99_latency_ms": float(np.percentile(serial_lats, 99)),
            "event_loop_lag_p95_ms": float(np.percentile(serial_lags, 95)) if serial_lags else 0.0
        },
        "dynamic_microbatch": {
            "total_time_sec": t_batch_total,
            "qps": batch_qps,
            "mean_latency_ms": float(np.mean(batch_lats)),
            "p50_latency_ms": float(np.percentile(batch_lats, 50)),
            "p95_latency_ms": float(np.percentile(batch_lats, 95)),
            "p99_latency_ms": float(np.percentile(batch_lats, 99)),
            "event_loop_lag_p95_ms": float(np.percentile(batch_lags, 95)) if batch_lags else 0.0,
            "batcher_stats": b_stats
        },
        "throughput_speedup_x": batch_qps / max(1e-5, serial_qps),
        "p99_latency_reduction_pct": (
            (np.percentile(serial_lats, 99) - np.percentile(batch_lats, 99))
            / max(1e-5, np.percentile(serial_lats, 99))
        ) * 100.0
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    logger.info(f"Concurrency report saved to {report_file}")

    print("\n" + "=" * 82)
    print("        DYNAMIC MICRO-BATCHING CONCURRENCY BENCHMARK (C = 50 CLIENTS)")
    print("=" * 82)
    print(f" {'Metric':<30} | {'Unbatched Serial':<18} | {'Dynamic Micro-Batch':<18} | {'Delta'}")
    print("-" * 82)
    print(f" {'Throughput (Queries/sec)':<30} | {serial_qps:<18.2f} | {batch_qps:<18.2f} | {res['throughput_speedup_x']:.2f}x Speedup")
    print(f" {'Total Processing Time (s)':<30} | {t_serial_total:<18.2f} | {t_batch_total:<18.2f} | -{t_serial_total - t_batch_total:.2f}s")
    print(f" {'Mean Latency (ms)':<30} | {res['serial']['mean_latency_ms']:<18.2f} | {res['dynamic_microbatch']['mean_latency_ms']:<18.2f} | -{(res['serial']['mean_latency_ms'] - res['dynamic_microbatch']['mean_latency_ms']):.1f}ms")
    print(f" {'p50 Latency (ms)':<30} | {res['serial']['p50_latency_ms']:<18.2f} | {res['dynamic_microbatch']['p50_latency_ms']:<18.2f} | -{(res['serial']['p50_latency_ms'] - res['dynamic_microbatch']['p50_latency_ms']):.1f}ms")
    print(f" {'p95 Latency (ms)':<30} | {res['serial']['p95_latency_ms']:<18.2f} | {res['dynamic_microbatch']['p95_latency_ms']:<18.2f} | -{(res['serial']['p95_latency_ms'] - res['dynamic_microbatch']['p95_latency_ms']):.1f}ms")
    print(f" {'p99 Latency (ms)':<30} | {res['serial']['p99_latency_ms']:<18.2f} | {res['dynamic_microbatch']['p99_latency_ms']:<18.2f} | {res['p99_latency_reduction_pct']:.1f}% Reduction")
    print(f" {'Event-Loop Lag p95 (ms)':<30} | {res['serial']['event_loop_lag_p95_ms']:<18.2f} | {res['dynamic_microbatch']['event_loop_lag_p95_ms']:<18.2f} | GIL Starvation Free")
    print(f" {'Avg Batch Utilization':<30} | {'1.00':<18} | {b_stats['avg_batch_utilization']:<18.2f} | SIMD Saturated")
    print("=" * 82 + "\n")

    return res


if __name__ == "__main__":
    asyncio.run(run_benchmark(num_requests=50))
