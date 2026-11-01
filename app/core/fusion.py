from typing import List, Dict, Tuple, Optional
from app.models.schemas import DocumentChunk, RetrievedChunk

class ReciprocalRankFusion:
    """
    Implements Cormack, Clarke, and Buettcher (2009) Reciprocal Rank Fusion (RRF)
    for combining dense semantic embeddings and sparse lexical BM25 rankings.
    """

    def __init__(self, default_k: int = 60):
        self.default_k = default_k

    def fuse(
        self,
        dense_results: List[Tuple[DocumentChunk, float, int]],  # (chunk, score, rank)
        sparse_results: List[Tuple[DocumentChunk, float, int]], # (chunk, score, rank)
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        rrf_k: Optional[int] = None,
        top_n: int = 10
    ) -> List[RetrievedChunk]:
        """
        Merge dense and sparse ranked lists using Reciprocal Rank Fusion.

        Formula:
            RRF_score(d) = (w_dense / (k + rank_dense(d))) + (w_sparse / (k + rank_sparse(d)))

        Args:
            dense_results: Ranked tuples from vector search (1-indexed ranks).
            sparse_results: Ranked tuples from BM25 (1-indexed ranks).
            dense_weight: Relative importance of dense retrieval.
            sparse_weight: Relative importance of sparse retrieval.
            rrf_k: Smoothing constant (default: 60).
            top_n: Number of fused results to return.

        Returns:
            List of RetrievedChunk sorted by fused RRF score descending.
        """
        k = rrf_k if rrf_k is not None else self.default_k

        # Aggregation maps by chunk ID
        chunks_map: Dict[str, DocumentChunk] = {}
        dense_info: Dict[str, Tuple[float, int]] = {}   # id -> (score, rank)
        sparse_info: Dict[str, Tuple[float, int]] = {}  # id -> (score, rank)
        rrf_scores: Dict[str, float] = {}

        # 1. Process dense results
        for chunk, score, rank in dense_results:
            chunks_map[chunk.id] = chunk
            dense_info[chunk.id] = (score, rank)
            rrf_contribution = dense_weight / (k + rank)
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + rrf_contribution

        # 2. Process sparse results
        for chunk, score, rank in sparse_results:
            if chunk.id not in chunks_map:
                chunks_map[chunk.id] = chunk
            sparse_info[chunk.id] = (score, rank)
            rrf_contribution = sparse_weight / (k + rank)
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + rrf_contribution

        # 3. Sort by fused RRF score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        # 4. Construct telemetry-rich RetrievedChunk instances
        fused_chunks: List[RetrievedChunk] = []
        for cid in sorted_ids[:top_n]:
            chunk = chunks_map[cid]
            d_score, d_rank = dense_info.get(cid, (None, None))
            s_score, s_rank = sparse_info.get(cid, (None, None))
            f_score = rrf_scores[cid]

            fused_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    dense_score=round(d_score, 4) if d_score is not None else None,
                    dense_rank=d_rank,
                    sparse_score=round(s_score, 4) if s_score is not None else None,
                    sparse_rank=s_rank,
                    rrf_score=round(f_score, 6),
                    rerank_score=None,
                    final_score=round(f_score, 6)
                )
            )

        return fused_chunks
