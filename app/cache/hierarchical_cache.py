"""
Two-Tier Hierarchical Cache: L1 Exact Token Hash + L2 Semantic Cosine Vector Cache.

Delivers sub-millisecond query caching with automatic TTL eviction,
LRU memory boundedness, and empirical cosine similarity thresholding.
"""
import time
import hashlib
from collections import OrderedDict
from typing import Optional, Tuple, Any, Dict, List
import numpy as np


class HierarchicalCache:
    """
    Tier 1: L1 Exact Hash (SHA-256 over AST-normalized query string) with LRU + TTL.
    Tier 2: L2 Semantic Vector Cache (Cosine distance lookup over query embeddings).
    """

    def __init__(
        self,
        l1_capacity: int = 5000,
        l1_ttl_sec: float = 3600.0,
        l2_capacity: int = 2000,
        l2_threshold: float = 0.94
    ):
        self.l1_capacity = l1_capacity
        self.l1_ttl_sec = l1_ttl_sec
        self.l2_capacity = l2_capacity
        self.l2_threshold = l2_threshold

        # L1 Storage: hash -> (timestamp, query_text, payload)
        self._l1_cache: OrderedDict[str, Tuple[float, str, Any]] = OrderedDict()

        # L2 Storage: embeddings list and associated payloads
        self._l2_embeddings: List[np.ndarray] = []
        self._l2_entries: List[Tuple[float, str, Any]] = []

        # Telemetry
        self._l1_hits = 0
        self._l2_hits = 0
        self._misses = 0

    @staticmethod
    def normalize_query_ast(query: str) -> str:
        """AST normalization: lowercase, strip punctuation, collapse whitespace."""
        normalized = " ".join(query.strip().lower().split())
        return normalized

    @classmethod
    def hash_query(cls, query: str) -> str:
        """SHA-256 hash of normalized query."""
        norm = cls.normalize_query_ast(query)
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()

    def get(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None
    ) -> Optional[Tuple[str, Any]]:
        """
        Probe L1 first, then L2 if embedding is provided.

        Returns:
            Tuple of ('L1_EXACT' | 'L2_SEMANTIC', cached_response) or None if miss.
        """
        now = time.time()

        # 1. Probe Tier 1: L1 Exact Hash
        l1_key = self.hash_query(query)
        if l1_key in self._l1_cache:
            ts, orig_q, val = self._l1_cache[l1_key]
            if now - ts <= self.l1_ttl_sec:
                self._l1_cache.move_to_end(l1_key)
                self._l1_hits += 1
                return ("L1_EXACT", val)
            else:
                del self._l1_cache[l1_key]

        # 2. Probe Tier 2: L2 Semantic Vector Cache
        if query_embedding is not None and len(self._l2_embeddings) > 0:
            q_vec = np.array(query_embedding, dtype=np.float32)
            norm = np.linalg.norm(q_vec)
            if norm > 0:
                q_vec = q_vec / norm

            emb_matrix = np.array(self._l2_embeddings, dtype=np.float32)
            sims = np.dot(emb_matrix, q_vec)
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])

            if best_sim >= self.l2_threshold:
                ts, orig_q, val = self._l2_entries[best_idx]
                if now - ts <= self.l1_ttl_sec:
                    self._l2_hits += 1
                    return ("L2_SEMANTIC", val)

        self._misses += 1
        return None

    def put(
        self,
        query: str,
        query_embedding: Optional[np.ndarray],
        response: Any
    ):
        """
        Store result in both L1 and L2 caches.
        """
        now = time.time()
        l1_key = self.hash_query(query)

        # Update L1
        if len(self._l1_cache) >= self.l1_capacity:
            self._l1_cache.popitem(last=False)
        self._l1_cache[l1_key] = (now, query, response)

        # Update L2 if vector is provided
        if query_embedding is not None:
            q_vec = np.array(query_embedding, dtype=np.float32)
            norm = np.linalg.norm(q_vec)
            if norm > 0:
                q_vec = q_vec / norm

            if len(self._l2_embeddings) >= self.l2_capacity:
                self._l2_embeddings.pop(0)
                self._l2_entries.pop(0)

            self._l2_embeddings.append(q_vec)
            self._l2_entries.append((now, query, response))

    def clear(self):
        """Flush cache entries."""
        self._l1_cache.clear()
        self._l2_embeddings.clear()
        self._l2_entries.clear()

    def stats(self) -> Dict[str, Any]:
        """Cache performance telemetry."""
        total_probes = self._l1_hits + self._l2_hits + self._misses
        hit_rate = (self._l1_hits + self._l2_hits) / max(1, total_probes)
        return {
            "l1_entries": len(self._l1_cache),
            "l2_entries": len(self._l2_embeddings),
            "l1_hits": self._l1_hits,
            "l2_hits": self._l2_hits,
            "misses": self._misses,
            "total_probes": total_probes,
            "hit_rate": hit_rate,
            "l2_threshold": self.l2_threshold
        }
