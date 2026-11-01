import re
import numpy as np
from typing import List, Dict, Any, Optional

class RAGMetrics:
    """
    Quantitative evaluation metrics aligned with RAG Triad & Ragas formulations:
    - Faithfulness (Context Grounding)
    - Context Precision@K
    - Context Recall
    - Hit Rate@K & Mean Reciprocal Rank (MRR)
    - Answer Relevancy
    - Latency (P50, P95, Mean)
    """

    @staticmethod
    def calculate_faithfulness(answer: str, retrieved_contexts: List[str]) -> float:
        """
        Measures the factual consistency of the answer against retrieved context.
        Formula: (Number of supported claims / Total claims in answer)
        """
        if not answer or not retrieved_contexts:
            return 0.0

        all_context = " ".join(retrieved_contexts).lower()

        # Split answer into statements/sentences
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer) if len(s.strip()) > 8]
        if not sentences:
            return 1.0

        supported_sentences = 0
        for sent in sentences:
            # Clean citations like [1], [2]
            clean_sent = re.sub(r'\[\d+\]', '', sent).strip().lower()
            key_terms = [w for w in re.findall(r'\b[a-zA-Z]{3,}\b', clean_sent)]
            if not key_terms:
                supported_sentences += 1
                continue

            # Check what proportion of content words appear in the retrieved context
            term_matches = sum(1 for term in key_terms if term in all_context)
            if term_matches / len(key_terms) >= 0.65:
                supported_sentences += 1

        return round(supported_sentences / len(sentences), 4)

    @staticmethod
    def is_chunk_relevant(chunk_text: str, ground_truth_fragments: List[str]) -> bool:
        """Determines if a chunk matches any ground truth reference context fragment."""
        chunk_lower = chunk_text.lower()
        for fragment in ground_truth_fragments:
            frag_lower = fragment.strip().lower()
            if not frag_lower:
                continue
            # Check substring match or high keyword overlap
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
        """
        Fraction of ground-truth fragments captured in the retrieved chunks.
        """
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
    def calculate_answer_relevancy(query: str, answer: str) -> float:
        """
        Quantifies how well the generated answer addresses the question.
        Uses token alignment and intent overlap.
        """
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
