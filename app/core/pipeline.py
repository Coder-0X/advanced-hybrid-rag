import time
import logging
from typing import List, Optional, Tuple, Dict, Any

from app.models.schemas import (
    DocumentInput,
    DocumentChunk,
    RetrievalMode,
    RetrievedChunk,
    QueryRequest,
    QueryResponse,
    LatencyBreakdown,
    IngestResponse
)
from app.core.document_processor import DocumentProcessor
from app.core.vector_store import DenseVectorStore
from app.core.bm25_retriever import BM25Retriever
from app.core.fusion import ReciprocalRankFusion
from app.core.reranker import CrossEncoderReranker
from app.core.generator import GroundedRAGGenerator
from app.config import settings

logger = logging.getLogger(__name__)

class HybridRAGEngine:
    """
    Unified RAG Engine supporting Naive (Dense), Hybrid (Dense + BM25 via RRF),
    and Reranked (Hybrid + Cross-Encoder) retrieval modes with latency telemetry.
    """

    def __init__(
        self,
        vector_store: Optional[DenseVectorStore] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        fusion: Optional[ReciprocalRankFusion] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        generator: Optional[GroundedRAGGenerator] = None,
        processor: Optional[DocumentProcessor] = None
    ):
        self.processor = processor or DocumentProcessor(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        self.vector_store = vector_store or DenseVectorStore(
            collection_name=settings.QDRANT_COLLECTION,
            in_memory=settings.USE_IN_MEMORY_DB
        )
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.fusion = fusion or ReciprocalRankFusion(default_k=settings.RRF_K)
        self.reranker = reranker or CrossEncoderReranker(model_name=settings.RERANKER_MODEL)
        self.generator = generator or GroundedRAGGenerator()

    def ingest_documents(self, documents: List[DocumentInput]) -> IngestResponse:
        """Process documents into chunks and index them into both vector and BM25 stores."""
        start_time = time.perf_counter()
        
        chunks = self.processor.process_batch(documents)
        if not chunks:
            return IngestResponse(
                status="empty",
                indexed_documents=len(documents),
                total_chunks=0,
                duration_ms=0.0
            )

        # Index dense embeddings
        self.vector_store.add_chunks(chunks)

        # Index sparse BM25
        self.bm25_retriever.add_chunks(chunks)

        duration = (time.perf_counter() - start_time) * 1000.0
        return IngestResponse(
            status="success",
            indexed_documents=len(documents),
            total_chunks=len(chunks),
            duration_ms=round(duration, 2)
        )

    def retrieve(
        self,
        query: str,
        mode: RetrievalMode = RetrievalMode.RERANKED,
        top_k: int = 5,
        dense_k: int = 10,
        sparse_k: int = 10,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        rrf_k: int = 60
    ) -> Tuple[List[RetrievedChunk], LatencyBreakdown]:
        """
        Execute information retrieval according to the specified mode, capturing granular latencies.
        """
        latencies = LatencyBreakdown()
        pipeline_start = time.perf_counter()

        if mode == RetrievalMode.NAIVE:
            # 1. Naive: Dense Vector Retrieval Only
            t0 = time.perf_counter()
            dense_results = self.vector_store.search(query, top_k=top_k)
            latencies.dense_retrieval_ms = round((time.perf_counter() - t0) * 1000.0, 2)

            retrieved = []
            for chunk, score, rank in dense_results:
                retrieved.append(
                    RetrievedChunk(
                        chunk_id=chunk.id,
                        document_id=chunk.document_id,
                        text=chunk.text,
                        metadata=chunk.metadata,
                        dense_score=round(score, 4),
                        dense_rank=rank,
                        sparse_score=None,
                        sparse_rank=None,
                        rrf_score=None,
                        rerank_score=None,
                        final_score=round(score, 4)
                    )
                )

        elif mode == RetrievalMode.HYBRID:
            # 2. Hybrid: Dense + Sparse BM25 merged via RRF
            t0 = time.perf_counter()
            dense_results = self.vector_store.search(query, top_k=dense_k)
            latencies.dense_retrieval_ms = round((time.perf_counter() - t0) * 1000.0, 2)

            t1 = time.perf_counter()
            sparse_results = self.bm25_retriever.retrieve(query, top_k=sparse_k)
            latencies.sparse_retrieval_ms = round((time.perf_counter() - t1) * 1000.0, 2)

            t2 = time.perf_counter()
            retrieved = self.fusion.fuse(
                dense_results=dense_results,
                sparse_results=sparse_results,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
                rrf_k=rrf_k,
                top_n=top_k
            )
            latencies.fusion_ms = round((time.perf_counter() - t2) * 1000.0, 2)

        elif mode == RetrievalMode.RERANKED:
            # 3. Reranked: Hybrid Search Candidates -> Cross-Encoder Reranking
            t0 = time.perf_counter()
            dense_results = self.vector_store.search(query, top_k=dense_k)
            latencies.dense_retrieval_ms = round((time.perf_counter() - t0) * 1000.0, 2)

            t1 = time.perf_counter()
            sparse_results = self.bm25_retriever.retrieve(query, top_k=sparse_k)
            latencies.sparse_retrieval_ms = round((time.perf_counter() - t1) * 1000.0, 2)

            # Generate candidate pool for reranker (e.g., top 15 candidates)
            candidate_pool_size = max(top_k * 3, 15)
            t2 = time.perf_counter()
            hybrid_candidates = self.fusion.fuse(
                dense_results=dense_results,
                sparse_results=sparse_results,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
                rrf_k=rrf_k,
                top_n=candidate_pool_size
            )
            latencies.fusion_ms = round((time.perf_counter() - t2) * 1000.0, 2)

            t3 = time.perf_counter()
            retrieved = self.reranker.rerank(
                query=query,
                candidates=hybrid_candidates,
                top_k=top_k
            )
            latencies.reranking_ms = round((time.perf_counter() - t3) * 1000.0, 2)

        else:
            raise ValueError(f"Unsupported retrieval mode: {mode}")

        latencies.total_pipeline_ms = round((time.perf_counter() - pipeline_start) * 1000.0, 2)
        return retrieved, latencies

    def query(self, req: QueryRequest) -> QueryResponse:
        """Execute full end-to-end RAG query: retrieval + generation."""
        retrieved, latencies = self.retrieve(
            query=req.query,
            mode=req.mode,
            top_k=req.top_k,
            dense_k=req.dense_k,
            sparse_k=req.sparse_k,
            dense_weight=req.dense_weight,
            sparse_weight=req.sparse_weight,
            rrf_k=req.rrf_k
        )

        answer = None
        grounding_score = None

        if req.generate_answer:
            t_gen = time.perf_counter()
            answer, grounding_score = self.generator.generate(req.query, retrieved)
            gen_ms = round((time.perf_counter() - t_gen) * 1000.0, 2)
            latencies.generation_ms = gen_ms
            latencies.total_pipeline_ms = round(latencies.total_pipeline_ms + gen_ms, 2)

        return QueryResponse(
            query=req.query,
            mode=req.mode,
            answer=answer,
            retrieved_chunks=retrieved,
            latencies=latencies,
            grounding_score=grounding_score
        )

    def stats(self) -> Dict[str, Any]:
        """Index diagnostics and statistics."""
        return {
            "total_chunks_vector_store": self.vector_store.count(),
            "total_chunks_bm25": self.bm25_retriever.total_documents,
            "embedding_model": settings.EMBEDDING_MODEL,
            "reranker_model": settings.RERANKER_MODEL,
            "in_memory_db": settings.USE_IN_MEMORY_DB
        }

    def clear(self) -> None:
        """Reset all indices."""
        self.vector_store.clear()
        self.bm25_retriever.clear()
