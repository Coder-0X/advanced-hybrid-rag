from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class RetrievalMode(str, Enum):
    NAIVE = "naive"       # Dense only
    HYBRID = "hybrid"     # Dense + BM25 via RRF
    RERANKED = "reranked" # Dense + BM25 via RRF -> Cross-Encoder

class DocumentInput(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DocumentChunk(BaseModel):
    id: str
    document_id: str
    chunk_index: int
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class IngestRequest(BaseModel):
    documents: List[DocumentInput]

class IngestResponse(BaseModel):
    status: str = "success"
    indexed_documents: int
    total_chunks: int
    duration_ms: float

class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Information Retrieval telemetry
    dense_score: Optional[float] = None
    dense_rank: Optional[int] = None
    sparse_score: Optional[float] = None
    sparse_rank: Optional[int] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    final_score: float

class LatencyBreakdown(BaseModel):
    dense_retrieval_ms: float = 0.0
    sparse_retrieval_ms: float = 0.0
    fusion_ms: float = 0.0
    reranking_ms: float = 0.0
    generation_ms: float = 0.0
    total_pipeline_ms: float = 0.0

class QueryRequest(BaseModel):
    query: str
    mode: RetrievalMode = RetrievalMode.RERANKED
    top_k: int = 5
    dense_k: int = 10
    sparse_k: int = 10
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    rrf_k: int = 60
    generate_answer: bool = True

class QueryResponse(BaseModel):
    query: str
    mode: RetrievalMode
    answer: Optional[str] = None
    retrieved_chunks: List[RetrievedChunk]
    latencies: LatencyBreakdown
    grounding_score: Optional[float] = None

class BenchmarkDatasetItem(BaseModel):
    id: str
    query: str
    ground_truth_answer: str
    ground_truth_context_fragments: List[str]
    target_document_id: Optional[str] = None
    category: Optional[str] = "general"

class MetricSummary(BaseModel):
    faithfulness: float = Field(..., description="Factuality/grounding of answer in retrieved context (0.0 to 1.0)")
    context_precision: float = Field(..., description="Are relevant chunks positioned at top ranks? (0.0 to 1.0)")
    context_recall: float = Field(..., description="Fraction of ground-truth context retrieved (0.0 to 1.0)")
    hit_rate: float = Field(..., description="Did retrieval contain the ground truth? (0.0 or 1.0 average)")
    mrr: float = Field(..., description="Mean Reciprocal Rank of first relevant chunk")
    answer_relevancy: float = Field(..., description="Semantic alignment of answer to query (0.0 to 1.0)")
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float

class PipelineBenchmarkComparison(BaseModel):
    naive: MetricSummary
    hybrid: MetricSummary
    reranked: MetricSummary
    delta_hybrid_vs_naive: Dict[str, float]
    delta_reranked_vs_naive: Dict[str, float]

class BenchmarkReport(BaseModel):
    run_id: str
    timestamp: str
    sample_count: int
    summary: PipelineBenchmarkComparison
    detailed_results: List[Dict[str, Any]]
