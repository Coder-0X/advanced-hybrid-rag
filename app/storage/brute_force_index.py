"""
Exact O(N*D) Brute-Force Vector Scan Index.

Serves as the absolute mathematical ground-truth baseline (G_q) for ANN benchmarking,
following standard SIGIR / VLDB evaluation protocols (Aumüller et al., ANN-Benchmarks).
"""
import time
from typing import List, Tuple, Set, Dict, Any, Optional
import numpy as np


class ExactBruteForceIndex:
    """
    Exhaustive linear scan matrix-multiplication vector index.
    Computes exact top-K cosine similarities with zero approximation error.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.doc_ids: List[str] = []
        self.vectors: Optional[np.ndarray] = None  # Shape: (N, D)

    def add(self, doc_id: str, vector: np.ndarray):
        """Add a single vector to the index."""
        vec = np.array(vector, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        if self.vectors is None:
            self.vectors = vec
        else:
            self.vectors = np.vstack([self.vectors, vec])
        self.doc_ids.append(doc_id)

    def add_batch(self, doc_ids: List[str], vectors: np.ndarray):
        """Add batch of vectors."""
        vecs = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        normalized_vecs = vecs / norms

        if self.vectors is None:
            self.vectors = normalized_vecs
        else:
            self.vectors = np.vstack([self.vectors, normalized_vecs])
        self.doc_ids.extend(doc_ids)

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Exhaustive matrix-vector dot product search.

        Returns:
            List of (doc_id, cosine_similarity) sorted descending.
        """
        if self.vectors is None or len(self.doc_ids) == 0:
            return []

        q_norm = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(q_norm)
        if norm > 0:
            q_norm = q_norm / norm

        # Exact cosine similarities via matrix multiplication: (1, D) x (D, N) -> (1, N)
        sims = np.dot(q_norm, self.vectors.T)[0]

        k = min(top_k, len(self.doc_ids))
        top_indices = np.argsort(-sims)[:k]

        return [(self.doc_ids[idx], float(sims[idx])) for idx in top_indices]

    def compute_ground_truth_sets(
        self,
        query_vectors: np.ndarray,
        top_k: int = 10
    ) -> List[Set[str]]:
        """
        Compute top-K ground-truth document ID sets for a matrix of queries.
        """
        ground_truth: List[Set[str]] = []
        for q_vec in query_vectors:
            results = self.search(q_vec, top_k=top_k)
            ground_truth.append({doc_id for doc_id, _ in results})
        return ground_truth

    def size(self) -> int:
        return len(self.doc_ids)
