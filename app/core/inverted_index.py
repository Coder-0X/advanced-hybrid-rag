"""
First-Principles Inverted Index and BM25+ (BM25Plus) Retrieval Engine.
Implements posting lists, document frequency tracking, length normalization,
and sparse score accumulation according to Information Retrieval literature
(Lv & Zhai, 2011; Robertson & Zaragoza, 2009).
"""
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

# Standard English IR stopword list
STOPWORDS: Set[str] = {
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

@dataclass(slots=True)
class Posting:
    """Represents an entry in an inverted posting list."""
    doc_index: int
    tf: int

class InvertedIndex:
    """
    First-principles Inverted Index supporting BM25+ (BM25Plus) ranking.
    Features:
      - Term-to-Postings inverted lists: Dict[str, List[Posting]]
      - Document length tracking and avgdl normalization
      - Sparse score accumulation (iterating only over postings of query terms)
      - BM25+ formulation with lower-bound parameter delta = 1.0 (Lv & Zhai, 2011)
        preventing over-penalization of long documents.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75, delta: float = 1.0):
        self.k1 = k1
        self.b = b
        self.delta = delta
        
        # Inverted index mappings
        self.postings: Dict[str, List[Posting]] = {}
        self.doc_lengths: List[int] = []
        self.doc_ids: List[str] = []
        
        # Corpus statistics
        self.total_docs: int = 0
        self.total_terms: int = 0
        self.avgdl: float = 0.0

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize text into alphanumeric terms, filtering English stopwords."""
        tokens = re.findall(r'\b[a-zA-Z0-9_\-\.]+\b', text.lower())
        return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

    def add_document(self, doc_id: str, text: str) -> int:
        """
        Tokenize and index a document.
        Returns the assigned internal integer index.
        """
        doc_idx = self.total_docs
        tokens = self.tokenize(text)
        doc_len = len(tokens)
        
        # Track doc metadata
        self.doc_ids.append(doc_id)
        self.doc_lengths.append(doc_len)
        self.total_terms += doc_len
        self.total_docs += 1
        self.avgdl = self.total_terms / self.total_docs if self.total_docs > 0 else 0.0

        # Compute within-document term frequencies
        tf_map: Dict[str, int] = {}
        for token in tokens:
            tf_map[token] = tf_map.get(token, 0) + 1

        # Append to posting lists
        for term, tf in tf_map.items():
            if term not in self.postings:
                self.postings[term] = []
            self.postings[term].append(Posting(doc_index=doc_idx, tf=tf))

        return doc_idx

    def get_idf(self, term: str) -> float:
        """
        Compute Robertson-Spärck Jones IDF with Robertson-Zaragoza smoothing:
          IDF(t) = ln((N - n(t) + 0.5) / (n(t) + 0.5) + 1.0)
        Guaranteed to be strictly positive.
        """
        if term not in self.postings:
            return 0.0
        df = len(self.postings[term])
        return math.log(1.0 + (self.total_docs - df + 0.5) / (df + 0.5))

    def score(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Retrieve and rank top_k document indices using BM25+.
        Uses sparse accumulator traversal: only visits documents containing at least
        one query token.
        
        Returns:
          List of (doc_index, score) pairs sorted descending by score.
        """
        if self.total_docs == 0:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            query_tokens = re.findall(r'\b\w+\b', query.lower())
            if not query_tokens:
                return []

        # Sparse accumulators for documents matching query terms
        scores: Dict[int, float] = {}

        for term in set(query_tokens):
            if term not in self.postings:
                continue

            idf = self.get_idf(term)
            if idf <= 0.0:
                continue

            # Traverse posting list for term
            for posting in self.postings[term]:
                doc_idx = posting.doc_index
                tf = posting.tf
                doc_len = self.doc_lengths[doc_idx]
                
                # Length normalization
                len_norm = 1.0 - self.b + self.b * (doc_len / self.avgdl if self.avgdl > 0 else 1.0)
                
                # BM25+ term score: idf * ((tf * (k1 + 1)) / (tf + k1 * len_norm) + delta)
                term_score = idf * (((tf * (self.k1 + 1.0)) / (tf + self.k1 * len_norm)) + self.delta)
                scores[doc_idx] = scores.get(doc_idx, 0.0) + term_score

        if not scores:
            return []

        # Sort top-k
        sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_docs[:top_k]

    def clear(self) -> None:
        """Reset the inverted index."""
        self.postings.clear()
        self.doc_lengths.clear()
        self.doc_ids.clear()
        self.total_docs = 0
        self.total_terms = 0
        self.avgdl = 0.0
