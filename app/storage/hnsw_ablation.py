"""
HNSW Graph Hyperparameter Ablation Study Engine.

Systematically benchmarks approximate nearest neighbor (ANN) graph parameters:
- M: Out-degree connectivity per node (M in [8, 16, 32, 64])
- ef_construct: Beam width during index building (ef_construct in [64, 128, 256])
- ef_search: Beam width during query traversal (ef_search in [10, 20, 50, 100, 200, 400])

Evaluates true empirical Recall@K against exact O(N*D) brute-force ground truth,
measuring QPS, P95 latency, build time, and index memory footprint.
"""
import time
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Set, Tuple, Any, Optional
import numpy as np
import psutil
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.storage.brute_force_index import ExactBruteForceIndex

logger = logging.getLogger(__name__)


@dataclass
class HNSWConfig:
    m: int
    ef_construct: int
    ef_search: int
    on_disk: bool = False


@dataclass
class AblationMetrics:
    m: int
    ef_construct: int
    ef_search: int
    build_time_sec: float
    index_size_mb: float
    qps: float
    p50_latency_ms: float
    p95_latency_ms: float
    recall_at_k: float


class HNSWAblationEngine:
    """
    Automated HNSW Pareto-frontier exploration harness.
    """

    def __init__(self, client: Optional[QdrantClient] = None, collection_prefix: str = "ablation"):
        self.client = client or QdrantClient(":memory:")
        self.prefix = collection_prefix

    def evaluate_configuration(
        self,
        config: HNSWConfig,
        corpus_ids: List[str],
        corpus_vectors: np.ndarray,
        query_vectors: np.ndarray,
        ground_truth: List[Set[str]],
        k: int = 10
    ) -> AblationMetrics:
        """
        Build index with (M, ef_construct), query with ef_search, and measure Recall@K against ground truth.
        """
        dim = corpus_vectors.shape[1]
        collection_name = f"{self.prefix}_m{config.m}_efc{config.ef_construct}"

        # 1. Create collection with specific HNSW graph parameters
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=dim,
                distance=models.Distance.COSINE,
                on_disk=config.on_disk
            ),
            hnsw_config=models.HnswConfigDiff(
                m=config.m,
                ef_construct=config.ef_construct,
                on_disk=config.on_disk
            )
        )

        # 2. Measure Index Build Latency & Resident Set Size Memory
        process = psutil.Process()
        mem_before = process.memory_info().rss
        t0 = time.perf_counter()

        batch_size = 64
        for i in range(0, len(corpus_ids), batch_size):
            batch_slice = slice(i, i + batch_size)
            batch_ids = corpus_ids[batch_slice]
            batch_vecs = corpus_vectors[batch_slice].tolist()

            points = [
                models.PointStruct(id=idx, vector=vec, payload={"doc_id": doc_id})
                for idx, (doc_id, vec) in enumerate(zip(batch_ids, batch_vecs), start=i)
            ]
            self.client.upsert(collection_name=collection_name, points=points)

        build_time = time.perf_counter() - t0
        mem_after = process.memory_info().rss
        index_size_mb = max(0.01, (mem_after - mem_before) / (1024 * 1024))

        # 3. Query Search Latency & Empirical Recall@K
        latencies = []
        hits = 0
        search_params = models.SearchParams(hnsw_ef=config.ef_search)

        for q_idx, q_vec in enumerate(query_vectors):
            t_start = time.perf_counter()
            if hasattr(self.client, "query_points"):
                q_res = self.client.query_points(
                    collection_name=collection_name,
                    query=q_vec.tolist(),
                    limit=k,
                    search_params=search_params
                )
                results = q_res.points
            else:
                results = self.client.search(
                    collection_name=collection_name,
                    query_vector=q_vec.tolist(),
                    limit=k,
                    search_params=search_params
                )
            lat = (time.perf_counter() - t_start) * 1000.0
            latencies.append(lat)

            retrieved_doc_ids = {r.payload.get("doc_id", str(r.id)) for r in results}
            gt_set = ground_truth[q_idx]
            overlap = len(retrieved_doc_ids.intersection(gt_set))
            hits += overlap

        total_possible = len(query_vectors) * k
        recall_at_k = hits / max(1, total_possible)
        total_search_time = sum(latencies) / 1000.0
        qps = len(query_vectors) / max(1e-5, total_search_time)
        p50_lat = float(np.percentile(latencies, 50))
        p95_lat = float(np.percentile(latencies, 95))

        # Cleanup collection
        try:
            self.client.delete_collection(collection_name=collection_name)
        except Exception:
            pass

        return AblationMetrics(
            m=config.m,
            ef_construct=config.ef_construct,
            ef_search=config.ef_search,
            build_time_sec=build_time,
            index_size_mb=index_size_mb,
            qps=qps,
            p50_latency_ms=p50_lat,
            p95_latency_ms=p95_lat,
            recall_at_k=recall_at_k
        )

    def run_sweep(
        self,
        corpus_ids: List[str],
        corpus_vectors: np.ndarray,
        query_vectors: np.ndarray,
        m_values: List[int] = [8, 16, 32, 64],
        ef_construct_values: List[int] = [64, 128],
        ef_search_values: List[int] = [10, 20, 50, 100, 200],
        k: int = 10
    ) -> List[AblationMetrics]:
        """
        Executes parameter sweep over (M, ef_construct, ef_search).
        """
        logger.info("Computing exact O(N*D) brute-force ground truth...")
        bf = ExactBruteForceIndex(dimension=corpus_vectors.shape[1])
        bf.add_batch(corpus_ids, corpus_vectors)
        ground_truth = bf.compute_ground_truth_sets(query_vectors, top_k=k)

        results: List[AblationMetrics] = []
        total_runs = len(m_values) * len(ef_construct_values) * len(ef_search_values)
        run_idx = 0

        logger.info(f"Starting HNSW grid sweep across {total_runs} parameter configurations...")

        for m in m_values:
            for ef_c in ef_construct_values:
                for ef_s in ef_search_values:
                    run_idx += 1
                    cfg = HNSWConfig(m=m, ef_construct=ef_c, ef_search=ef_s)
                    metrics = self.evaluate_configuration(
                        config=cfg,
                        corpus_ids=corpus_ids,
                        corpus_vectors=corpus_vectors,
                        query_vectors=query_vectors,
                        ground_truth=ground_truth,
                        k=k
                    )
                    results.append(metrics)
                    logger.info(
                        f"[{run_idx}/{total_runs}] M={m:2d} ef_c={ef_c:3d} ef_s={ef_s:3d} | "
                        f"Recall@{k}={metrics.recall_at_k:.4f} | QPS={metrics.qps:7.1f} | "
                        f"P95={metrics.p95_latency_ms:5.2f}ms | Build={metrics.build_time_sec:5.2f}s"
                    )

        return results
