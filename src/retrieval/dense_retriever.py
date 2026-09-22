from typing import List, Optional
from ..rag.schemas import RetrievedChunk
from ..vectorstore.qdrant_store import QdrantVectorStore


class DenseRetriever:
    """Semantic vector retriever using BGE-M3 embeddings and Qdrant."""

    def __init__(self, vectorstore: Optional[QdrantVectorStore] = None):
        self.vectorstore = vectorstore or QdrantVectorStore()

    def retrieve(self, query: str, top_k: int = 10, query_vector: Optional[List[float]] = None) -> List[RetrievedChunk]:
        return self.vectorstore.search_dense(query=query, top_k=top_k, query_vector=query_vector)

