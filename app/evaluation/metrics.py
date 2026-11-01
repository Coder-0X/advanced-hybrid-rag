"""
Quantitative Evaluation Metrics for Information Retrieval and RAG Benchmarking.
Implements:
  - Cranfield Graded NDCG@K (Discounted Cumulative Gain with 2^rel - 1 gain)
  - Cranfield MAP@K (Mean Average Precision normalized by R_q)
  - Mean Reciprocal Rank (MRR@K) and Hit Rate@K
  - Context Precision@K and Context Recall
  - Reference-based ROUGE-L (Longest Common Subsequence F1) and Token F1
  - Latency percentiles (P50, P95, Mean)
"""
import math
import re
from typing import Any, Dict, List, Optional, Set
import numpy as np

class RAGMetrics:
    """
    Comprehensive Information Retrieval & Generation Evaluation Suite.
    """

    # -------------------------------------------------------------
    # Cranfield Information Retrieval Ranking Metrics (SIGIR / TREC)
    # -------------------------------------------------------------

    @staticmethod
    def calculate_ndcg_at_k(
        retrieved_doc_ids: List[str],
        qrels_for_query: Dict[str, int],
        k: int = 10
    ) -> float:
        """
        Graded Normalized Discounted Cumulative Gain (NDCG@K).
        Deduplicates passage-level retrieved doc IDs preserving first rank appearance,
        ensuring DCG <= IDCG and strict mathematical bounds [0.0, 1.0].
        Formula:
          DCG@K = sum_{i=1}^K (2^{rel_i} - 1) / log_2(i + 1)
          IDCG@K = sum_{i=1}^{min(K, |R|)} (2^{rel^*_i} - 1) / log_2(i + 1)
          NDCG@K = DCG@K / IDCG@K
        """
        if not retrieved_doc_ids or not qrels_for_query or k <= 0:
            return 0.0

        # Deduplicate retrieved document IDs preserving rank order
        seen_docs: Set[str] = set()
        unique_docs: List[str] = []
        for doc_id in retrieved_doc_ids:
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                unique_docs.append(doc_id)

        top_docs = unique_docs[:k]

        # Calculate DCG@K
        dcg = 0.0
        for i, doc_id in enumerate(top_docs):
            rel = qrels_for_query.get(doc_id, 0)
            if rel > 0:
                dcg += (2.0 ** rel - 1.0) / math.log2(i + 2)

        # Calculate Ideal DCG (IDCG@K)
        all_relevances = sorted(
            [grade for grade in qrels_for_query.values() if grade > 0],
            reverse=True
        )
        if not all_relevances:
            return 0.0

        idcg = 0.0
        for i, rel in enumerate(all_relevances[:k]):
            idcg += (2.0 ** rel - 1.0) / math.log2(i + 2)

        if idcg == 0.0:
            return 0.0

        return round(min(1.0, dcg / idcg), 4)

    @staticmethod
    def calculate_map_at_k(
        retrieved_doc_ids: List[str],
        qrels_for_query: Dict[str, int],
        k: int = 10
    ) -> float:
        """
        Cranfield Mean Average Precision at K (MAP@K).
        Deduplicates retrieved passage documents to document-level evaluation.
        Normalized by the total number of relevant documents R_q in qrels:
          AP@K = (1 / min(K, R_q)) * sum_{i=1}^K (Precision@i * is_rel(i))
        """
        if not retrieved_doc_ids or not qrels_for_query or k <= 0:
            return 0.0

        r_q = sum(1 for grade in qrels_for_query.values() if grade > 0)
        if r_q == 0:
            return 0.0

        # Deduplicate retrieved document IDs preserving rank order
        seen_docs: Set[str] = set()
        unique_docs: List[str] = []
        for doc_id in retrieved_doc_ids:
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                unique_docs.append(doc_id)

        top_docs = unique_docs[:k]
        running_rel = 0
        precisions = []

        for i, doc_id in enumerate(top_docs, start=1):
            if qrels_for_query.get(doc_id, 0) > 0:
                running_rel += 1
                precisions.append(running_rel / i)

        if not precisions:
            return 0.0

        denom = min(k, r_q)
        return round(min(1.0, sum(precisions) / denom), 4)

    # -------------------------------------------------------------
    # Reference-Based Generation Metrics (ROUGE-L, Token F1)
    # -------------------------------------------------------------

    @staticmethod
    def calculate_rouge_l(candidate: str, reference: str) -> float:
        """
        Calculates reference-based ROUGE-L (Longest Common Subsequence F1 score)
        between candidate answer and gold-standard reference.
        """
        cand_tokens = re.findall(r'\b\w+\b', candidate.lower())
        ref_tokens = re.findall(r'\b\w+\b', reference.lower())

        m, n = len(cand_tokens), len(ref_tokens)
        if m == 0 or n == 0:
            return 0.0

        # Dynamic programming for LCS
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if cand_tokens[i - 1] == ref_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs_len = dp[m][n]
        if lcs_len == 0:
            return 0.0

        precision = lcs_len / m
        recall = lcs_len / n
        f1 = (2.0 * precision * recall) / (precision + recall)
        return round(f1, 4)

    @staticmethod
    def calculate_token_f1(candidate: str, reference: str) -> float:
        """Computes token-level precision, recall, and harmonic F1 score."""
        cand_tokens = re.findall(r'\b\w+\b', candidate.lower())
        ref_tokens = re.findall(r'\b\w+\b', reference.lower())

        if not cand_tokens or not ref_tokens:
            return 0.0

        from collections import Counter
        cand_counts = Counter(cand_tokens)
        ref_counts = Counter(ref_tokens)

        overlap = sum((cand_counts & ref_counts).values())
        if overlap == 0:
            return 0.0

        precision = overlap / len(cand_tokens)
        recall = overlap / len(ref_tokens)
        f1 = (2.0 * precision * recall) / (precision + recall)
        return round(f1, 4)

    # -------------------------------------------------------------
    # Passage Retrieval & Context Metrics
    # -------------------------------------------------------------

    @staticmethod
    def is_chunk_relevant(chunk_text: str, ground_truth_fragments: List[str]) -> bool:
        """Determines if a chunk matches any ground truth reference context fragment."""
        chunk_lower = chunk_text.lower()
        for fragment in ground_truth_fragments:
            frag_lower = fragment.strip().lower()
            if not frag_lower:
                continue
            if frag_lower in chunk_lower:
                return True
            frag_words = set(re.findall(r'\b\w{3,}\b', frag_lower))
            if frag_words:
                chunk_words = set(re.findall(r'\b\w{3,}\b', chunk_lower))
                overlap = len(frag_words.intersection(chunk_words)) / len(frag_words)
                if overlap >= 0.70:
                    return True
        return False

    @classmethod
    def calculate_context_precision(
        cls,
        retrieved_chunks: List[str],
        ground_truth_fragments: List[str]
    ) -> float:
        """
        Evaluates whether relevant chunks are placed at higher ranks.
        Formula: sum_{k=1}^K (Precision@k * is_relevant(k)) / total_relevant_in_top_k
        """
        if not retrieved_chunks or not ground_truth_fragments:
            return 0.0

        relevance_flags = [
            1 if cls.is_chunk_relevant(chunk, ground_truth_fragments) else 0
            for chunk in retrieved_chunks
        ]

        total_relevant = sum(relevance_flags)
        if total_relevant == 0:
            return 0.0

        precisions_at_k = []
        running_relevant = 0
        for k, flag in enumerate(relevance_flags, start=1):
            if flag == 1:
                running_relevant += 1
                precisions_at_k.append(running_relevant / k)

        return round(sum(precisions_at_k) / total_relevant, 4)

    @classmethod
    def calculate_context_recall(
        cls,
        retrieved_chunks: List[str],
        ground_truth_fragments: List[str]
    ) -> float:
        """Fraction of ground-truth fragments captured in the retrieved chunks."""
        if not ground_truth_fragments:
            return 1.0
        if not retrieved_chunks:
            return 0.0

        all_retrieved = " ".join(retrieved_chunks).lower()
        matched = 0
        for frag in ground_truth_fragments:
            frag_lower = frag.strip().lower()
            if frag_lower in all_retrieved:
                matched += 1
            else:
                frag_words = set(re.findall(r'\b\w{3,}\b', frag_lower))
                if frag_words:
                    all_words = set(re.findall(r'\b\w{3,}\b', all_retrieved))
                    if len(frag_words.intersection(all_words)) / len(frag_words) >= 0.70:
                        matched += 1

        return round(matched / len(ground_truth_fragments), 4)

    @classmethod
    def calculate_hit_rate(
        cls,
        retrieved_chunks: List[str],
        ground_truth_fragments: List[str]
    ) -> float:
        """1.0 if at least one relevant passage was retrieved, else 0.0."""
        for chunk in retrieved_chunks:
            if cls.is_chunk_relevant(chunk, ground_truth_fragments):
                return 1.0
        return 0.0

    @classmethod
    def calculate_mrr(
        cls,
        retrieved_chunks: List[str],
        ground_truth_fragments: List[str]
    ) -> float:
        """Reciprocal rank of the first relevant retrieved chunk."""
        for rank, chunk in enumerate(retrieved_chunks, start=1):
            if cls.is_chunk_relevant(chunk, ground_truth_fragments):
                return round(1.0 / rank, 4)
        return 0.0

    @staticmethod
    def calculate_faithfulness(answer: str, retrieved_contexts: List[str]) -> float:
        """
        Measures the factual consistency of the answer against retrieved context.
        Formula: (Number of supported claims / Total claims in answer)
        """
        if not answer or not retrieved_contexts:
            return 0.0

        all_context = " ".join(retrieved_contexts).lower()

        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer) if len(s.strip()) > 8]
        if not sentences:
            return 1.0

        supported_sentences = 0
        for sent in sentences:
            clean_sent = re.sub(r'\[\d+\]', '', sent).strip().lower()
            key_terms = [w for w in re.findall(r'\b[a-zA-Z]{3,}\b', clean_sent)]
            if not key_terms:
                supported_sentences += 1
                continue

            term_matches = sum(1 for term in key_terms if term in all_context)
            if term_matches / len(key_terms) >= 0.65:
                supported_sentences += 1

        return round(supported_sentences / len(sentences), 4)

    @staticmethod
    def calculate_answer_relevancy(query: str, answer: str) -> float:
        """Quantifies how well the generated answer addresses the question."""
        if not answer:
            return 0.0
        q_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()))
        a_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', answer.lower()))
        if not q_words:
            return 1.0

        overlap = len(q_words.intersection(a_words))
        score = min(1.0, (overlap / len(q_words)) * 1.2)
        return round(score, 4)

    @staticmethod
    def calculate_latency_stats(latencies_ms: List[float]) -> Dict[str, float]:
        """Compute mean, P50 (median), and P95 latency."""
        if not latencies_ms:
            return {"mean": 0.0, "p50": 0.0, "p95": 0.0}

        arr = np.array(latencies_ms)
        return {
            "mean": round(float(np.mean(arr)), 2),
            "p50": round(float(np.percentile(arr, 50)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2)
        }
