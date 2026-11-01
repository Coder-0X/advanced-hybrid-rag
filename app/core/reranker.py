import math
import logging
from typing import List, Optional
from app.models.schemas import RetrievedChunk
from app.config import settings

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    """
    Reranks candidate chunks using a deep cross-encoder model to capture
    full query-document cross-attention and filter out spurious retrievals.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.RERANKER_MODEL
        self._model = None
        self._init_model()

    def _init_model(self):
        """Initialize sentence-transformers CrossEncoder or fallback."""
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading CrossEncoder reranker: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
            logger.info("CrossEncoder reranker loaded successfully.")
        except Exception as e:
            logger.warning(f"CrossEncoder model unavailable ({e}). Using exact-match cross-attention fallback.")
            self._model = _FallbackCrossEncoder()

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Sigmoid activation to map unbounded logits to [0.0, 1.0] probability."""
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def rerank(self, query: str, candidates: List[RetrievedChunk], top_k: int = 5) -> List[RetrievedChunk]:
        """
        Rerank candidate chunks using the cross-encoder.

        Args:
            query: User search query.
            candidates: Candidate RetrievedChunk instances (e.g. from RRF hybrid search).
            top_k: Number of highest-relevance chunks to return.

        Returns:
            List of reranked RetrievedChunk instances sorted by rerank_score descending.
        """
        if not candidates:
            return []

        # Form (query, passage) pairs
        pairs = [(query, c.text) for c in candidates]

        if hasattr(self._model, "predict"):
            raw_scores = self._model.predict(pairs)
        else:
            raw_scores = self._model.score_pairs(pairs)

        scored_candidates = []
        for chunk, raw_score in zip(candidates, raw_scores):
            # If model outputs logits, map to probability via sigmoid; if already [0, 1], preserve
            score_val = float(raw_score)
            normalized_score = self._sigmoid(score_val) if (score_val < 0 or score_val > 1.0) else score_val

            # Create a new updated chunk copy
            updated_chunk = chunk.model_copy()
            updated_chunk.rerank_score = round(normalized_score, 4)
            updated_chunk.final_score = round(normalized_score, 4)
            scored_candidates.append(updated_chunk)

        # Sort descending by cross-encoder score
        scored_candidates.sort(key=lambda c: c.final_score, reverse=True)

        return scored_candidates[:top_k]


class _FallbackCrossEncoder:
    """Robust heuristic reranker combining token overlap, bigram alignment, and phrase coverage."""
    def score_pairs(self, pairs: List[tuple]) -> List[float]:
        scores = []
        for query, text in pairs:
            q_lower = query.lower()
            t_lower = text.lower()
            q_words = [w for w in q_lower.split() if len(w) > 2]
            
            if not q_words:
                scores.append(0.5)
                continue

            # Exact phrase match boost
            exact_match_boost = 0.4 if q_lower in t_lower else 0.0

            # Token overlap coverage
            matched_words = sum(1 for w in q_words if w in t_lower)
            coverage = matched_words / len(q_words)

            # Bigram overlap
            q_bigrams = [f"{q_words[i]} {q_words[i+1]}" for i in range(len(q_words)-1)]
            bigram_boost = 0.0
            if q_bigrams:
                matched_bigrams = sum(1 for bg in q_bigrams if bg in t_lower)
                bigram_boost = 0.3 * (matched_bigrams / len(q_bigrams))

            score = 0.3 * coverage + exact_match_boost + bigram_boost
            scores.append(min(1.0, score))
        return scores
