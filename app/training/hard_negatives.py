"""
Hard Negative Mining Module for Neural Ranking Distillation.

Extracts adversarial lexical and dense negative passages from first-stage retrieval
(BM25+ and Dense Vector Search) that yield zero ground-truth relevance (TREC qrels == 0).
Mined triplets (query, pos_passage, neg_passage) force the student Cross-Encoder
to capture fine-grained domain distinctions rather than distinguishing random distractors.
"""
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Triplet:
    query: str
    positive_text: str
    negative_text: str
    query_id: str
    pos_doc_id: str
    neg_doc_id: str


@dataclass
class TeacherScoredTriplet:
    query: str
    positive_text: str
    negative_text: str
    query_id: str
    pos_doc_id: str
    neg_doc_id: str
    teacher_pos_score: float
    teacher_neg_score: float
    teacher_margin: float


class HardNegativeMiner:
    """
    Identifies hard false-positive negatives from BM25+ and dense vector retrieval pools.
    Constructs high-information triplets for MarginMSE and InfoNCE loss optimization.
    """

    def __init__(self, qrels: Dict[str, Dict[str, int]]):
        """
        Args:
            qrels: Ground-truth relevance mapping: {query_id: {doc_id: relevance_grade}}
        """
        self.qrels = qrels

    def mine_triplets_for_query(
        self,
        query_id: str,
        query_text: str,
        retrieved_bm25_ids: List[str],
        retrieved_dense_ids: List[str],
        corpus_lookup: Dict[str, str],
        num_negatives: int = 4,
        candidate_depth: int = 50,
        random_seed: int = 42
    ) -> List[Triplet]:
        """
        Mine hard negative triplets for a single query.

        Args:
            query_id: Query identifier.
            query_text: Natural language query string.
            retrieved_bm25_ids: Ranked list of document/chunk IDs from BM25+.
            retrieved_dense_ids: Ranked list of document/chunk IDs from Dense vector search.
            corpus_lookup: Mapping from doc/chunk ID to text content.
            num_negatives: Number of hard negatives to pair per positive document.
            candidate_depth: Search horizon in first-stage results.
            random_seed: Seed for reproducible negative sampling.

        Returns:
            List of Triplet instances.
        """
        rng = np.random.default_rng(random_seed)
        q_rel = self.qrels.get(query_id, {})
        positive_ids = [doc_id for doc_id, rel in q_rel.items() if rel > 0]

        if not positive_ids:
            return []

        # Candidate pool: top-K candidates from both retrievers
        candidate_set = []
        for doc_id in retrieved_bm25_ids[:candidate_depth] + retrieved_dense_ids[:candidate_depth]:
            if doc_id not in candidate_set:
                candidate_set.append(doc_id)

        # Filter strictly for candidates where ground-truth relevance is 0
        hard_negative_ids = [
            doc_id for doc_id in candidate_set
            if q_rel.get(doc_id, 0) == 0 and doc_id in corpus_lookup
        ]

        if not hard_negative_ids:
            # Fallback to any non-positive in corpus if candidates are exhausted
            all_non_pos = [k for k in corpus_lookup if q_rel.get(k, 0) == 0]
            if all_non_pos:
                hard_negative_ids = list(rng.choice(
                    all_non_pos,
                    size=min(num_negatives, len(all_non_pos)),
                    replace=False
                ))
            else:
                return []

        triplets: List[Triplet] = []
        for pos_id in positive_ids:
            pos_text = corpus_lookup.get(pos_id)
            if not pos_text:
                continue

            sample_size = min(num_negatives, len(hard_negative_ids))
            sampled_negs = rng.choice(hard_negative_ids, size=sample_size, replace=False)

            for neg_id in sampled_negs:
                neg_text = corpus_lookup.get(neg_id)
                if not neg_text:
                    continue

                triplets.append(
                    Triplet(
                        query=query_text,
                        positive_text=pos_text,
                        negative_text=neg_text,
                        query_id=query_id,
                        pos_doc_id=pos_id,
                        neg_doc_id=neg_id
                    )
                )

        return triplets

    def mine_corpus_triplets(
        self,
        engine: Any,
        queries: List[Any],
        num_negatives: int = 4,
        candidate_depth: int = 50
    ) -> List[Triplet]:
        """
        Execute live retrieval against the engine to extract mined hard negatives across queries.

        Args:
            engine: Initialized HybridRAGEngine instance.
            queries: List of BenchmarkDatasetItem or dict-like queries with '_id' and 'text'.
            num_negatives: Negatives to sample per positive document.
            candidate_depth: Horizon of first-stage retrieval candidates.

        Returns:
            List of Triplet instances ready for training.
        """
        # Build corpus lookup from engine
        corpus_lookup: Dict[str, str] = {}
        for chunk_id, chunk in engine.vector_store.chunks_cache.items():
            corpus_lookup[chunk_id] = chunk.text
            # Also map parent document_id
            if chunk.document_id not in corpus_lookup:
                corpus_lookup[chunk.document_id] = chunk.text

        all_triplets: List[Triplet] = []

        for q_idx, item in enumerate(queries):
            q_id = getattr(item, "_id", None) or getattr(item, "id", None) or item.get("_id") or item.get("id")
            q_text = getattr(item, "text", None) or getattr(item, "query", None) or item.get("text") or item.get("query")

            if not q_id or not q_text:
                continue

            # First-stage retrieval
            dense_results = engine.vector_store.search(q_text, top_k=candidate_depth)
            sparse_results = engine.bm25_index.search(q_text, top_k=candidate_depth)

            bm25_ids = [doc_id for doc_id, _ in sparse_results]
            dense_ids = [chunk.document_id for chunk, _ in dense_results]

            triplets = self.mine_triplets_for_query(
                query_id=q_id,
                query_text=q_text,
                retrieved_bm25_ids=bm25_ids,
                retrieved_dense_ids=dense_ids,
                corpus_lookup=corpus_lookup,
                num_negatives=num_negatives,
                candidate_depth=candidate_depth,
                random_seed=42 + q_idx
            )
            all_triplets.extend(triplets)

        logger.info(f"Mined {len(all_triplets)} hard-negative triplets across {len(queries)} queries.")
        return all_triplets

    @staticmethod
    def score_with_teacher(
        triplets: List[Triplet],
        teacher_model: Any,
        batch_size: int = 32
    ) -> List[TeacherScoredTriplet]:
        """
        Score (query, positive) and (query, negative) pairs with a Cross-Encoder teacher
        to generate continuous margin targets: Delta_T = T(q, pos) - T(q, neg).

        Args:
            triplets: Unscored mined triplets.
            teacher_model: SentenceTransformers CrossEncoder instance.
            batch_size: Inference batch size.

        Returns:
            List of TeacherScoredTriplet instances with teacher predictions.
        """
        pos_pairs = [[t.query, t.positive_text] for t in triplets]
        neg_pairs = [[t.query, t.negative_text] for t in triplets]

        pos_scores = teacher_model.predict(pos_pairs, batch_size=batch_size, show_progress_bar=False)
        neg_scores = teacher_model.predict(neg_pairs, batch_size=batch_size, show_progress_bar=False)

        scored: List[TeacherScoredTriplet] = []
        for t, s_pos, s_neg in zip(triplets, pos_scores, neg_scores):
            margin = float(s_pos - s_neg)
            scored.append(
                TeacherScoredTriplet(
                    query=t.query,
                    positive_text=t.positive_text,
                    negative_text=t.negative_text,
                    query_id=t.query_id,
                    pos_doc_id=t.pos_doc_id,
                    neg_doc_id=t.neg_doc_id,
                    teacher_pos_score=float(s_pos),
                    teacher_neg_score=float(s_neg),
                    teacher_margin=margin
                )
            )
        return scored
