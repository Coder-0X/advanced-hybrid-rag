import re
import math
from typing import List, Dict, Tuple, Optional
from app.models.schemas import DocumentChunk

# Common English stopwords for high-quality lexical retrieval
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
    "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more",
    "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's",
    "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they",
    "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't",
    "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom",
    "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
    "your", "yours", "yourself", "yourselves"
}

class BM25Retriever:
    """Sparse retrieval engine utilizing the Okapi BM25 ranking function."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[DocumentChunk] = []
        self.chunk_id_to_chunk: Dict[str, DocumentChunk] = {}
        self.corpus_tokens: List[List[str]] = []
        self._bm25_model = None

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize text into lowercase alphanumeric words, filtering out common stopwords."""
        tokens = re.findall(r'\b[a-zA-Z0-9_\-\.]+\b', text.lower())
        return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Add new document chunks to the BM25 index and refresh statistics."""
        for chunk in chunks:
            if chunk.id not in self.chunk_id_to_chunk:
                self.chunks.append(chunk)
                self.chunk_id_to_chunk[chunk.id] = chunk
                self.corpus_tokens.append(self.tokenize(chunk.text))

        self._build_index()

    def _build_index(self) -> None:
        """Build or update the BM25 index."""
        if not self.corpus_tokens:
            self._bm25_model = None
            return

        try:
            from rank_bm25 import BM25Plus
            # BM25Plus prevents zero or negative IDF on small corpora
            self._bm25_model = BM25Plus(self.corpus_tokens, k1=self.k1, b=self.b)
        except ImportError:
            try:
                from rank_bm25 import BM25Okapi
                self._bm25_model = BM25Okapi(self.corpus_tokens, k1=self.k1, b=self.b)
            except ImportError:
                self._bm25_model = _NativeBM25Okapi(self.corpus_tokens, k1=self.k1, b=self.b)

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[DocumentChunk, float, int]]:
        """
        Retrieve top_k chunks matching the query.
        Returns: List of (DocumentChunk, score, rank) tuples, 1-indexed rank.
        """
        if not self.chunks or not self._bm25_model:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            # Fallback to raw words if query was all stopwords
            query_tokens = re.findall(r'\b\w+\b', query.lower())
            if not query_tokens:
                return []

        scores = self._bm25_model.get_scores(query_tokens)
        
        # Sort indices by score descending
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results: List[Tuple[DocumentChunk, float, int]] = []
        rank = 1
        for idx in ranked_indices[:top_k]:
            score = float(scores[idx])
            # Even if score is 0, we provide rank up to top_k if corpus is small, but track score
            results.append((self.chunks[idx], score, rank))
            rank += 1

        return results

    def clear(self) -> None:
        self.chunks.clear()
        self.chunk_id_to_chunk.clear()
        self.corpus_tokens.clear()
        self._bm25_model = None

    @property
    def total_documents(self) -> int:
        return len(self.chunks)


class _NativeBM25Okapi:
    """Lightweight pure-python BM25Okapi implementation for redundancy."""
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size > 0 else 0
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}

        df: Dict[str, int] = {}
        for doc in corpus:
            freq: Dict[str, int] = {}
            for token in doc:
                freq[token] = freq.get(token, 0) + 1
            self.doc_freqs.append(freq)
            for token in freq:
                df[token] = df.get(token, 0) + 1

        for token, freq in df.items():
            # Standard Lucene/BM25 IDF formula
            self.idf[token] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def get_scores(self, query: List[str]) -> List[float]:
        scores = [0.0] * self.corpus_size
        for token in query:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]
            for i, doc_freq in enumerate(self.doc_freqs):
                if token in doc_freq:
                    tf = doc_freq[token]
                    d_len = self.doc_len[i]
                    num = tf * (self.k1 + 1)
                    denom = tf + self.k1 * (1 - self.b + self.b * (d_len / self.avgdl if self.avgdl > 0 else 1.0))
                    scores[i] += idf_val * (num / denom)
        return scores
