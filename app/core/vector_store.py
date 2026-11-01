import uuid
import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter

from app.models.schemas import DocumentChunk
from app.config import settings

logger = logging.getLogger(__name__)

class DenseVectorStore:
    """
    Dense vector indexing and similarity retrieval utilizing Qdrant (in-memory or local storage).
    Integrates with SentenceTransformers embeddings with graceful fallback.
    """

    def __init__(
        self,
        collection_name: str = "rag_chunks",
        vector_size: int = 384,
        in_memory: bool = True,
        storage_path: Optional[str] = None
    ):
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.in_memory = in_memory
        self.storage_path = storage_path

        # Initialize Qdrant Client
        if self.in_memory:
            self.client = QdrantClient(":memory:")
        else:
            path = storage_path or str(settings.STORAGE_PATH)
            self.client = QdrantClient(path=path)

        self.chunks_cache: Dict[str, DocumentChunk] = {}
        self._embedder = None
        self._init_embedder()
        self._init_collection()

    def _init_embedder(self):
        """Initialize sentence-transformers model or fallback deterministic embedder."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            self._embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
            test_vec = self._embedder.encode("test", convert_to_numpy=True)
            self.vector_size = len(test_vec)
            logger.info(f"Embedding model loaded successfully. Vector dim: {self.vector_size}")
        except Exception as e:
            logger.warning(f"SentenceTransformer not yet available ({e}). Using robust semantic hashing fallback embedder.")
            self._embedder = _FallbackEmbedder(dim=self.vector_size)

    def _init_collection(self):
        """Create or ensure existence of Qdrant collection."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
            )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Encode a list of text strings into normalized dense vectors."""
        if not texts:
            return []
        if hasattr(self._embedder, "encode"):
            embeddings = self._embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return [vec.tolist() for vec in embeddings]
        return self._embedder.embed(texts)

    def embed_query(self, query: str) -> List[float]:
        """Encode a single query string."""
        return self.embed_texts([query])[0]

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Embed and upsert document chunks into Qdrant."""
        if not chunks:
            return

        texts = [c.text for c in chunks]
        vectors = self.embed_texts(texts)

        points = []
        for chunk, vector in zip(chunks, vectors):
            self.chunks_cache[chunk.id] = chunk
            # Qdrant requires UUID or integer ID for point ID
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.id))
            payload = {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "metadata": chunk.metadata
            }
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(self, query: str, top_k: int = 10) -> List[Tuple[DocumentChunk, float, int]]:
        """
        Execute dense cosine similarity search for the given query.
        Returns: List of (DocumentChunk, score, rank) tuples, 1-indexed rank.
        """
        if not self.chunks_cache:
            return []

        query_vector = self.embed_query(query)
        
        # Qdrant client 1.20+ uses query_points
        if hasattr(self.client, "query_points"):
            query_res = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k
            )
            hits = query_res.points
        else:
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k
            )

        results: List[Tuple[DocumentChunk, float, int]] = []
        for rank_idx, hit in enumerate(hits, start=1):
            chunk_id = hit.payload.get("chunk_id")
            chunk = self.chunks_cache.get(chunk_id)
            if not chunk:
                # Reconstruct if not found in memory cache
                chunk = DocumentChunk(
                    id=chunk_id,
                    document_id=hit.payload.get("document_id"),
                    chunk_index=hit.payload.get("chunk_index", 0),
                    text=hit.payload.get("text", ""),
                    metadata=hit.payload.get("metadata", {})
                )
            results.append((chunk, float(hit.score), rank_idx))

        return results

    def clear(self) -> None:
        """Clear the collection and chunk cache."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._init_collection()
        self.chunks_cache.clear()

    def count(self) -> int:
        """Total chunks in the vector store."""
        return len(self.chunks_cache)


class _FallbackEmbedder:
    """Deterministic hashing-based n-gram embedding vectorizer for fallback/testing."""
    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        results = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            words = text.lower().split()
            for w in words:
                h = hash(w) % self.dim
                vec[h] += 1.0
            # N-grams of characters
            for i in range(len(text) - 2):
                tri = text[i:i+3].lower()
                h = hash(tri) % self.dim
                vec[h] += 0.5
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            results.append(vec.tolist())
        return results
