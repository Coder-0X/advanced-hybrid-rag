"""
HNSW Pareto-Frontier Exploration & Publication Figure Generator.

Performs grid sweep across (M, ef_construct, ef_search) against exact O(N*D) ground truth.
Outputs reports/hnsw_ablation_results.json and publication-ready Pareto plot:
reports/figures/hnsw_pareto_frontier.png
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt

from app.core.pipeline import HybridRAGEngine
from app.evaluation.dataset import DEFAULT_CORPUS, DEFAULT_BENCHMARK_DATASET
from app.storage.hnsw_ablation import HNSWAblationEngine, AblationMetrics
from app.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hnsw_ablation_benchmark")


def run_hnsw_ablation_benchmark(output_json: str = "reports/hnsw_ablation_results.json"):
    fig_dir = settings.PROJECT_ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    report_file = settings.PROJECT_ROOT / output_json

    # 1. Initialize engine to extract pre-indexed vectors
    logger.info("Initializing HybridRAGEngine to encode corpus and queries...")
    engine = HybridRAGEngine()
    if engine.vector_store.count() == 0:
        engine.ingest_documents(DEFAULT_CORPUS)

    # Extract corpus vectors
    corpus_ids = []
    corpus_vec_list = []
    for chunk_id, chunk in engine.vector_store.chunks_cache.items():
        corpus_ids.append(chunk.id)
        # Re-encode or retrieve vector
        v = engine.vector_store._embedder.encode(chunk.text, convert_to_numpy=True)
        corpus_vec_list.append(v)

    corpus_vectors = np.array(corpus_vec_list, dtype=np.float32)

    # Encode test queries
    queries = [item.query for item in DEFAULT_BENCHMARK_DATASET[:35]]  # Representative query sample
    logger.info(f"Encoding {len(queries)} evaluation queries...")
    query_vectors = engine.vector_store._embedder.encode(queries, convert_to_numpy=True)

    # 2. Execute HNSW Parameter Sweep
    engine_ablation = HNSWAblationEngine()
    metrics_list = engine_ablation.run_sweep(
        corpus_ids=corpus_ids,
        corpus_vectors=corpus_vectors,
        query_vectors=query_vectors,
        m_values=[8, 16, 32, 64],
        ef_construct_values=[64, 128],
        ef_search_values=[10, 20, 50, 100, 200],
        k=10
    )

    # 3. Save JSON Report
    results_data = [m.__dict__ for m in metrics_list]
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    logger.info(f"Ablation results saved to {report_file}")

    # 4. Generate Publication Pareto Frontier Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Recall@10 vs QPS (Pareto Curve)
    color_map = {8: "#3b82f6", 16: "#10b981", 32: "#f59e0b", 64: "#ef4444"}

    for m in [8, 16, 32, 64]:
        subset = [m_res for m_res in metrics_list if m_res.m == m and m_res.ef_construct == 128]
        subset = sorted(subset, key=lambda x: x.recall_at_k)
        recalls = [x.recall_at_k for x in subset]
        qps_vals = [x.qps for x in subset]
        ax1.plot(recalls, qps_vals, marker="o", label=f"M={m} (ef_c=128)", color=color_map[m], linewidth=2)

    ax1.set_title("HNSW Pareto Frontier: Recall@10 vs Query Throughput", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Recall@10 vs Exact Brute-Force Ground Truth", fontsize=10)
    ax1.set_ylabel("Query Throughput (QPS)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower left")

    # Plot 2: Recall@10 vs Build Time
    for m in [8, 16, 32, 64]:
        subset = [m_res for m_res in metrics_list if m_res.m == m and m_res.ef_search == 100]
        subset = sorted(subset, key=lambda x: x.recall_at_k)
        recalls = [x.recall_at_k for x in subset]
        build_times = [x.build_time_sec * 1000 for x in subset]
        ax2.scatter(recalls, build_times, label=f"M={m}", color=color_map[m], s=80, alpha=0.8)

    ax2.set_title("HNSW Graph Connectivity vs Index Build Overhead", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Recall@10 vs Ground Truth", fontsize=10)
    ax2.set_ylabel("Index Build Time (ms)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    fig_path = fig_dir / "hnsw_pareto_frontier.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    logger.info(f"Pareto plot saved to {fig_path}")
    print(f"\n[SUCCESS] Generated HNSW Pareto Frontier: {fig_path}")


if __name__ == "__main__":
    run_hnsw_ablation_benchmark()
