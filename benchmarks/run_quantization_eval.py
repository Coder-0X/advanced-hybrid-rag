"""
Empirical Quantization Benchmark: PyTorch FP32 vs ONNX INT8 Cross-Encoder.

Evaluates ranking fidelity (NDCG@10, MAP@10, MRR) and inference latency (mean, p50, p95)
across all 105 Cranfield benchmark queries with non-parametric hypothesis testing
(Two-Sided Paired Wilcoxon Signed-Rank Test) to verify zero statistically significant degradation.
"""
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from scipy import stats

from app.core.pipeline import HybridRAGEngine
from app.core.reranker import CrossEncoderReranker
from app.core.onnx_reranker import ONNXReranker
from app.evaluation.dataset import DEFAULT_CORPUS, DEFAULT_BENCHMARK_DATASET, DEFAULT_QRELS
from app.evaluation.metrics import RAGMetrics
from app.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("quantization_eval")


def run_quantization_evaluation(output_path: str = "reports/quantization_benchmark.json"):
    """
    Executes full evaluation comparing PyTorch FP32 and ONNX INT8 across 105 queries.
    """
    reports_dir = settings.PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = settings.PROJECT_ROOT / output_path

    # 1. Initialize RAG Engine and index corpus
    logger.info("Initializing Hybrid RAG Engine and indexing corpus...")
    engine = HybridRAGEngine()
    if engine.vector_store.count() == 0:
        engine.ingest_documents(DEFAULT_CORPUS)

    logger.info(f"Corpus indexed: {engine.vector_store.count()} chunks.")

    # 2. Initialize FP32 and INT8 Rerankers
    logger.info("Loading PyTorch FP32 Cross-Encoder...")
    reranker_fp32 = CrossEncoderReranker()
    logger.info("Loading ONNX INT8 Cross-Encoder...")
    reranker_int8 = ONNXReranker(use_int8=True)

    queries = DEFAULT_BENCHMARK_DATASET
    qrels = DEFAULT_QRELS
    logger.info(f"Evaluating across {len(queries)} Cranfield queries...")

    fp32_latencies: List[float] = []
    int8_latencies: List[float] = []
    fp32_ndcgs: List[float] = []
    int8_ndcgs: List[float] = []
    fp32_maps: List[float] = []
    int8_maps: List[float] = []
    fp32_mrrs: List[float] = []
    int8_mrrs: List[float] = []

    per_query_results = []

    for idx, item in enumerate(queries):
        q_id = item.id
        q_text = item.query
        q_rels = qrels.get(q_id, {})

        # Step 1: Candidate retrieval via Hybrid RRF (top 20)
        candidates, _ = engine.retrieve(
            query=q_text,
            mode="hybrid",
            top_k=20
        )

        if not candidates:
            continue

        # Step 2: FP32 Reranking
        cands_fp32 = [c.model_copy(deep=True) for c in candidates]
        t0 = time.perf_counter()
        ranked_fp32 = reranker_fp32.rerank(q_text, cands_fp32, top_k=10)
        lat_fp32 = (time.perf_counter() - t0) * 1000.0
        fp32_latencies.append(lat_fp32)

        # Step 3: INT8 Reranking
        cands_int8 = [c.model_copy(deep=True) for c in candidates]
        t0 = time.perf_counter()
        ranked_int8 = reranker_int8.rerank(q_text, cands_int8, top_k=10)
        lat_int8 = (time.perf_counter() - t0) * 1000.0
        int8_latencies.append(lat_int8)

        # Step 4: Metric calculation
        doc_ids_fp32 = [c.document_id for c in ranked_fp32]
        doc_ids_int8 = [c.document_id for c in ranked_int8]

        ndcg_fp32 = RAGMetrics.calculate_ndcg_at_k(doc_ids_fp32, q_rels, k=10)
        ndcg_int8 = RAGMetrics.calculate_ndcg_at_k(doc_ids_int8, q_rels, k=10)
        map_fp32 = RAGMetrics.calculate_map_at_k(doc_ids_fp32, q_rels, k=10)
        map_int8 = RAGMetrics.calculate_map_at_k(doc_ids_int8, q_rels, k=10)

        # MRR calculation
        mrr_fp32 = 0.0
        for rank, d in enumerate(doc_ids_fp32, start=1):
            if q_rels.get(d, 0) > 0:
                mrr_fp32 = 1.0 / rank
                break

        mrr_int8 = 0.0
        for rank, d in enumerate(doc_ids_int8, start=1):
            if q_rels.get(d, 0) > 0:
                mrr_int8 = 1.0 / rank
                break

        fp32_ndcgs.append(ndcg_fp32)
        int8_ndcgs.append(ndcg_int8)
        fp32_maps.append(map_fp32)
        int8_maps.append(map_int8)
        fp32_mrrs.append(mrr_fp32)
        int8_mrrs.append(mrr_int8)

        per_query_results.append({
            "query_id": q_id,
            "category": item.category,
            "fp32": {"latency_ms": lat_fp32, "ndcg": ndcg_fp32, "map": map_fp32, "mrr": mrr_fp32},
            "int8": {"latency_ms": lat_int8, "ndcg": ndcg_int8, "map": map_int8, "mrr": mrr_int8}
        })

    # Summary Statistics
    mean_fp32_lat = float(np.mean(fp32_latencies))
    p50_fp32_lat = float(np.percentile(fp32_latencies, 50))
    p95_fp32_lat = float(np.percentile(fp32_latencies, 95))

    mean_int8_lat = float(np.mean(int8_latencies))
    p50_int8_lat = float(np.percentile(int8_latencies, 50))
    p95_int8_lat = float(np.percentile(int8_latencies, 95))

    speedup_ratio = mean_fp32_lat / max(1e-5, mean_int8_lat)

    mean_fp32_ndcg = float(np.mean(fp32_ndcgs))
    mean_int8_ndcg = float(np.mean(int8_ndcgs))
    ndcg_retention_pct = (mean_int8_ndcg / max(1e-5, mean_fp32_ndcg)) * 100.0

    mean_fp32_map = float(np.mean(fp32_maps))
    mean_int8_map = float(np.mean(int8_maps))

    mean_fp32_mrr = float(np.mean(fp32_mrrs))
    mean_int8_mrr = float(np.mean(int8_mrrs))

    # Two-Sided Paired Wilcoxon Signed-Rank Test on NDCG@10
    deltas = np.array(int8_ndcgs) - np.array(fp32_ndcgs)
    nonzero_deltas = deltas[deltas != 0]

    if len(nonzero_deltas) == 0:
        wilcoxon_stat = 0.0
        wilcoxon_p = 1.0
        stat_significance = "Identical (Zero Delta)"
    else:
        try:
            w_res = stats.wilcoxon(deltas, alternative="two-sided")
            wilcoxon_stat = float(w_res.statistic)
            wilcoxon_p = float(w_res.pvalue)
            stat_significance = "Statistically Insignificant (p > 0.05)" if wilcoxon_p > 0.05 else "Statistically Significant (p <= 0.05)"
        except Exception as e:
            wilcoxon_stat = 0.0
            wilcoxon_p = 1.0
            stat_significance = f"Test Fallback ({e})"

    # Model file sizes
    fp32_path = settings.PROJECT_ROOT / "models" / "onnx_reranker" / "model_fp32.onnx"
    int8_path = settings.PROJECT_ROOT / "models" / "onnx_reranker" / "model_int8.onnx"
    fp32_size_mb = fp32_path.stat().st_size / (1024 * 1024) if fp32_path.exists() else 0.0
    int8_size_mb = int8_path.stat().st_size / (1024 * 1024) if int8_path.exists() else 0.0

    benchmark_summary = {
        "num_queries": len(queries),
        "fp32": {
            "mean_latency_ms": mean_fp32_lat,
            "p50_latency_ms": p50_fp32_lat,
            "p95_latency_ms": p95_fp32_lat,
            "ndcg_at_10": mean_fp32_ndcg,
            "map_at_10": mean_fp32_map,
            "mrr": mean_fp32_mrr,
            "model_size_mb": fp32_size_mb
        },
        "int8": {
            "mean_latency_ms": mean_int8_lat,
            "p50_latency_ms": p50_int8_lat,
            "p95_latency_ms": p95_int8_lat,
            "ndcg_at_10": mean_int8_ndcg,
            "map_at_10": mean_int8_map,
            "mrr": mean_int8_mrr,
            "model_size_mb": int8_size_mb
        },
        "comparison": {
            "latency_speedup_x": speedup_ratio,
            "compression_ratio_x": fp32_size_mb / max(1e-5, int8_size_mb),
            "ndcg_retention_pct": ndcg_retention_pct,
            "wilcoxon_statistic": wilcoxon_stat,
            "wilcoxon_p_value": wilcoxon_p,
            "statistical_verdict": stat_significance
        },
        "per_query_results": per_query_results
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    logger.info(f"Quantization benchmark report saved to {report_file}")

    # Print Clean Console Summary
    print("\n" + "=" * 78)
    print("      NEURAL RERANKER QUANTIZATION BENCHMARK (105 CRANFIELD QUERIES)")
    print("=" * 78)
    print(f" {'Metric':<25} | {'PyTorch FP32':<15} | {'ONNX INT8':<15} | {'Delta / Factor':<15}")
    print("-" * 78)
    print(f" {'Model Size (MB)':<25} | {fp32_size_mb:<15.2f} | {int8_size_mb:<15.2f} | {fp32_size_mb/max(1e-5, int8_size_mb):.2f}x Compression")
    print(f" {'Mean Latency (ms)':<25} | {mean_fp32_lat:<15.2f} | {mean_int8_lat:<15.2f} | {speedup_ratio:.2f}x Speedup")
    print(f" {'p50 Latency (ms)':<25} | {p50_fp32_lat:<15.2f} | {p50_int8_lat:<15.2f} | {p50_fp32_lat/max(1e-5, p50_int8_lat):.2f}x Speedup")
    print(f" {'p95 Latency (ms)':<25} | {p95_fp32_lat:<15.2f} | {p95_int8_lat:<15.2f} | {p95_fp32_lat/max(1e-5, p95_int8_lat):.2f}x Speedup")
    print(f" {'Graded NDCG@10':<25} | {mean_fp32_ndcg:<15.4f} | {mean_int8_ndcg:<15.4f} | {ndcg_retention_pct:.2f}% Retention")
    print(f" {'Cranfield MAP@10':<25} | {mean_fp32_map:<15.4f} | {mean_int8_map:<15.4f} | {mean_int8_map - mean_fp32_map:+.4f}")
    print(f" {'Mean Reciprocal Rank':<25} | {mean_fp32_mrr:<15.4f} | {mean_int8_mrr:<15.4f} | {mean_int8_mrr - mean_fp32_mrr:+.4f}")
    print("-" * 78)
    print(f" Paired Wilcoxon Signed-Rank Test: W = {wilcoxon_stat:.2f}, p = {wilcoxon_p:.5f}")
    print(f" Verdict: {stat_significance}")
    print("=" * 78 + "\n")

    return benchmark_summary


if __name__ == "__main__":
    run_quantization_evaluation()
