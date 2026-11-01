import pytest
from app.core.fusion import ReciprocalRankFusion
from app.models.schemas import DocumentChunk

def test_rrf_scoring_logic():
    fusion = ReciprocalRankFusion(default_k=60)
    
    chunk_a = DocumentChunk(id="chunk_a", document_id="doc_a", chunk_index=0, text="Text A")
    chunk_b = DocumentChunk(id="chunk_b", document_id="doc_b", chunk_index=0, text="Text B")
    chunk_c = DocumentChunk(id="chunk_c", document_id="doc_c", chunk_index=0, text="Text C")

    # Dense ranking: A (rank 1), B (rank 2)
    dense_results = [(chunk_a, 0.95, 1), (chunk_b, 0.85, 2)]
    # Sparse ranking: B (rank 1), C (rank 2)
    sparse_results = [(chunk_b, 12.5, 1), (chunk_c, 8.2, 2)]

    fused = fusion.fuse(
        dense_results=dense_results,
        sparse_results=sparse_results,
        dense_weight=0.5,
        sparse_weight=0.5,
        rrf_k=60,
        top_n=3
    )

    # Chunk B is present in both dense (rank 2) and sparse (rank 1):
    # RRF(B) = 0.5 / (60 + 2) + 0.5 / (60 + 1) = 0.5/62 + 0.5/61 ~= 0.008064 + 0.008196 = 0.01626
    # Chunk A: 0.5 / (60 + 1) ~= 0.008196
    # Chunk C: 0.5 / (60 + 2) ~= 0.008064
    # Therefore, Chunk B MUST be ranked #1!
    assert len(fused) == 3
    assert fused[0].chunk_id == "chunk_b"
    assert fused[0].dense_rank == 2
    assert fused[0].sparse_rank == 1
    assert fused[0].final_score > fused[1].final_score

    assert fused[1].chunk_id == "chunk_a"
    assert fused[2].chunk_id == "chunk_c"
