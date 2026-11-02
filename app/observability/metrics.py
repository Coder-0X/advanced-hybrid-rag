"""
Prometheus Metrics Engine with Bounded Categorical Cardinality.

Tracks retrieval, reranking, and cache stage latencies with customized exponential
histogram buckets and bounded labels, completely eliminating cardinality explosion hazards.
"""
from typing import Optional, Tuple
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
    REGISTRY
)

# 1. Retrieval Latency Histogram across pipeline stages
RETRIEVAL_LATENCY_SECONDS = Histogram(
    "rag_retrieval_latency_seconds",
    "Latency of individual RAG retrieval stages in seconds",
    labelnames=["stage", "archetype"],
    buckets=(0.002, 0.005, 0.010, 0.025, 0.050, 0.075, 0.100, 0.250, 0.500, 1.0, 2.5)
)

# 2. Total requests counter
RETRIEVAL_REQUESTS_TOTAL = Counter(
    "rag_retrieval_requests_total",
    "Total retrieval operations executed",
    labelnames=["stage", "status"]
)

# 3. Candidate Pool Size Histogram
RETRIEVED_CHUNKS_COUNT = Histogram(
    "rag_retrieved_chunks_count",
    "Count of candidate chunks processed at each stage",
    labelnames=["stage"],
    buckets=(1, 3, 5, 10, 20, 30, 50, 100)
)

# 4. Hierarchical Cache Telemetry
CACHE_PROBES_TOTAL = Counter(
    "rag_cache_probes_total",
    "Cache lookup results partitioned by cache tier",
    labelnames=["tier", "result"]  # tier: 'L1_exact' | 'L2_semantic', result: 'hit' | 'miss'
)

# 5. Live in-flight requests gauge
IN_FLIGHT_REQUESTS = Gauge(
    "rag_in_flight_requests",
    "Current active query requests being processed"
)


def record_stage_latency(stage: str, duration_sec: float, archetype: str = "general"):
    """Record latency in seconds with bounded categorical labels."""
    RETRIEVAL_LATENCY_SECONDS.labels(stage=stage, archetype=archetype).observe(duration_sec)


def get_prometheus_metrics() -> Tuple[bytes, str]:
    """Generates Prometheus text representation and content type."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
