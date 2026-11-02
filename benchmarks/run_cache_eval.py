"""
Semantic Cache & Anti-Stampede SingleFlight Evaluation Benchmark.

1. Anti-Stampede SingleFlight Concurrency Test:
   Dispatches N = 50 concurrent identical requests to verify that exactly 1 downstream
   neural pipeline execution occurs while 49 requests are coalesced onto the active promise.
2. ROC & Youden's J Threshold Calibration:
   Sweeps cosine similarity thresholds across semantic paraphrase pairs and hard distractors,
   computing optimal operational threshold tau* maximizing Youden's J statistic.
"""
import asyncio
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt

from app.core.pipeline import HybridRAGEngine
from app.cache.single_flight import SingleFlight
from app.cache.hierarchical_cache import HierarchicalCache
from app.cache.threshold_evaluator import SemanticCacheThresholdCalibrator
from app.evaluation.dataset import DEFAULT_CORPUS, DEFAULT_BENCHMARK_DATASET
from app.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cache_eval")


async def run_stampede_stress_test(num_concurrent: int = 50) -> Dict[str, Any]:
    """
    Simulates high-concurrency query flood on an expensive retrieval pipeline
    with and without SingleFlight promise coalescing.
    """
    logger.info(f"Running SingleFlight Anti-Stampede test with N = {num_concurrent} concurrent callers...")
    
    execution_counter = 0

    async def expensive_downstream_retrieval(query: str):
        nonlocal execution_counter
        execution_counter += 1
        # Simulate neural pipeline latency (15ms)
        await asyncio.sleep(0.015)
        return {"query": query, "result": "sample retrieved chunks"}

    sf = SingleFlight()
    target_query = "What is the canonical Cormack smoothing constant k in Reciprocal Rank Fusion?"

    # Scenario: Concurrent dispatch through SingleFlight
    t0 = time.perf_counter()
    tasks = [
        sf.execute(target_query, lambda: expensive_downstream_retrieval(target_query))
        for _ in range(num_concurrent)
    ]
    results = await asyncio.gather(*tasks)
    total_time_ms = (time.perf_counter() - t0) * 1000.0

    sf_stats = sf.stats()
    logger.info(
        f"SingleFlight executed: {execution_counter} downstream run(s) for {num_concurrent} callers "
        f"({sf_stats['coalesced_calls']} coalesced, time: {total_time_ms:.1f}ms)."
    )

    return {
        "num_concurrent_requests": num_concurrent,
        "downstream_executions": execution_counter,
        "coalesced_requests": sf_stats["coalesced_calls"],
        "suppression_ratio": sf_stats["suppression_ratio"],
        "total_burst_latency_ms": total_time_ms
    }


def run_threshold_calibration() -> Tuple[float, Dict[str, Any]]:
    """
    Evaluates ROC curve across cosine similarity thresholds using real embeddings.
    """
    logger.info("Initializing embedding model for semantic threshold calibration...")
    engine = HybridRAGEngine()
    embedder = engine.vector_store._embedder

    # Pairs with semantic equivalence (Label = 1)
    positive_pairs = [
        ("What is the Cormack constant in RRF?", "What smoothing constant k does Cormack RRF use?"),
        ("How does HPA scale replicas?", "What is the formula used by Horizontal Pod Autoscaler?"),
        ("PostgreSQL tuple headers xmin and xmax", "Transaction visibility headers in Postgres MVCC"),
        ("Raft leader election split vote", "How does Raft handle partitioned elections?"),
        ("BM25+ delta parameter role", "Why does BM25+ add lower bound delta?")
    ]

    # Hard lexical distractors with differing intent (Label = 0)
    negative_pairs = [
        ("What is the Cormack constant in RRF?", "How does Kubernetes Ingress configure RRF?"),
        ("How does HPA scale replicas?", "How does Vertical Pod Autoscaler update limits?"),
        ("PostgreSQL tuple headers xmin and xmax", "PostgreSQL write-ahead logging checkpoints"),
        ("Raft leader election split vote", "Paxos consensus multi-decree proposer rules"),
        ("BM25+ delta parameter role", "Qdrant HNSW vector distance metric cosine")
    ]

    sims = []
    labels = []

    for q1, q2 in positive_pairs:
        v1 = embedder.encode(q1, convert_to_numpy=True)
        v2 = embedder.encode(q2, convert_to_numpy=True)
        cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        sims.append(cos)
        labels.append(1)

    for q1, q2 in negative_pairs:
        v1 = embedder.encode(q1, convert_to_numpy=True)
        v2 = embedder.encode(q2, convert_to_numpy=True)
        cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        sims.append(cos)
        labels.append(0)

    sims_arr = np.array(sims)
    labels_arr = np.array(labels)

    optimal_tau, best_pt, roc_points = SemanticCacheThresholdCalibrator.find_optimal_threshold(
        cosine_similarities=sims_arr,
        ground_truth_labels=labels_arr
    )

    logger.info(
        f"Optimal threshold calibrated: tau* = {optimal_tau:.2f} "
        f"(Youden's J = {best_pt.youden_j:.2f}, F1 = {best_pt.f1:.2f}, TPR = {best_pt.tpr:.2f}, FPR = {best_pt.fpr:.2f})"
    )

    # Plot ROC and Youden's J curves
    fig_dir = settings.PROJECT_ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "cache_threshold_calibration.png"

    thresholds = [pt.threshold for pt in roc_points]
    tprs = [pt.tpr for pt in roc_points]
    fprs = [pt.fpr for pt in roc_points]
    js = [pt.youden_j for pt in roc_points]

    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, tprs, label="True Positive Rate (Cache Hit Rate)", color="#10b981", linewidth=2)
    plt.plot(thresholds, fprs, label="False Positive Rate (Poisoning Risk)", color="#ef4444", linewidth=2)
    plt.plot(thresholds, js, label="Youden's J Statistic (TPR - FPR)", color="#3b82f6", linewidth=2.5, linestyle="--")
    plt.axvline(x=optimal_tau, color="#8b5cf6", linestyle=":", label=f"Optimal tau* = {optimal_tau:.2f}")

    plt.title("Semantic L2 Cache Threshold Calibration (ROC Analysis)", fontsize=12, fontweight="bold")
    plt.xlabel("Cosine Similarity Threshold (tau)", fontsize=10)
    plt.ylabel("Rate / Score", fontsize=10)
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    return optimal_tau, {
        "optimal_threshold_tau": optimal_tau,
        "best_roc_point": best_pt.__dict__,
        "roc_curve": [pt.__dict__ for pt in roc_points],
        "figure_path": str(fig_path)
    }


async def main():
    reports_dir = settings.PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / "cache_benchmark.json"

    stampede_res = await run_stampede_stress_test(num_concurrent=50)
    tau_star, calib_res = run_threshold_calibration()

    combined_report = {
        "single_flight_stampede_suppression": stampede_res,
        "threshold_calibration": calib_res
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(combined_report, f, indent=2)

    logger.info(f"Cache benchmark report saved to {report_file}")

    print("\n" + "=" * 80)
    print("      SEMANTIC CACHE & ANTI-STAMPEDE SINGLEFLIGHT BENCHMARK")
    print("=" * 80)
    print(f" Concurrent Duplicate Requests:    {stampede_res['num_concurrent_requests']}")
    print(f" Actual Downstream Executions:     {stampede_res['downstream_executions']} (Expected: 1)")
    print(f" Stampede Coalesced Requests:      {stampede_res['coalesced_requests']}")
    print(f" Stampede Suppression Ratio:       {stampede_res['suppression_ratio']*100:.1f}%")
    print(f" Calibrated Semantic Threshold:    tau* = {tau_star:.2f}")
    print(f" Max Youden's J Statistic:         {calib_res['best_roc_point']['youden_j']:.2f}")
    print(f" Cache Hit Rate at tau*:           {calib_res['best_roc_point']['tpr']*100:.1f}%")
    print(f" False Positive Poisoning Rate:    {calib_res['best_roc_point']['fpr']*100:.1f}%")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
