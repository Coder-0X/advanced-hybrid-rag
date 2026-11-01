"""
BM25 Retriever powered by first-principles InvertedIndex (BM25+).
Provides sparse lexical retrieval over document chunks with exact term inverted postings.
"""
from typing import Dict, List, Optional, Tuple
from app.core.inverted_index import InvertedIndex, STOPWORDS
from app.models.schemas import DocumentChunk

class BM25Retriever:
    """
    Sparse lexical retrieval engine utilizing a first-principles Inverted Index
    and BM25+ (Lv & Zhai, 2011) ranking function.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, delta: float = 1.0):
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.chunks: List[DocumentChunk] = []
        self.chunk_id_to_chunk: Dict[str, DocumentChunk] = {}
        self.index: InvertedIndex = InvertedIndex(k1=self.k1, b=self.b, delta=self.delta)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize text into lowercase alphanumeric words, filtering out common stopwords."""
        return InvertedIndex.tokenize(text)

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Add new document chunks to the sparse inverted index."""
        for chunk in chunks:
            if chunk.id not in self.chunk_id_to_chunk:
                self.chunks.append(chunk)
                self.chunk_id_to_chunk[chunk.id] = chunk
                self.index.add_document(doc_id=chunk.id, text=chunk.text)

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[DocumentChunk, float, int]]:
        """
        Retrieve top_k chunks matching the query.
        Returns: List of (DocumentChunk, score, rank) tuples, 1-indexed rank.
        """
        if not self.chunks or self.index.total_docs == 0:
            return []

        ranked_indices = self.index.score(query=query, top_k=top_k)

        results: List[Tuple[DocumentChunk, float, int]] = []
        for rank, (doc_idx, score) in enumerate(ranked_indices, start=1):
            results.append((self.chunks[doc_idx], float(score), rank))

        return results

    def clear(self) -> None:
        """Reset the retriever and underlying inverted index."""
        self.chunks.clear()
        self.chunk_id_to_chunk.clear()
        self.index.clear()

    @property
    def total_documents(self) -> int:
        return len(self.chunks)
