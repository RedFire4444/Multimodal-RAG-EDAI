import os
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from ..rag.schemas import Chunk, RetrievedChunk
from ..embeddings.bge_embeddings import SentenceTransformerEmbeddings

load_dotenv()


class QdrantVectorStore:
    """Manages Qdrant collection creation, indexing, payload storage, and similarity search.
    Supports Dense vectors (sentence-transformers/all-minilm-l6-v2) and Sparse BM25 (qdrant/bm25)
    with local embeddings fallback.
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        location: Optional[str] = None,
        cloud_inference: Optional[bool] = None,
        dense_model_name: Optional[str] = None,
        sparse_model_name: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        embeddings: Optional[SentenceTransformerEmbeddings] = None,
    ):
        self.collection_name = collection_name or os.getenv(
            "QDRANT_COLLECTION_NAME", "MultimodalRAG"
        )
        self.url = url or os.getenv("QDRANT_URL")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.location = location

        env_cloud_inf = os.getenv("QDRANT_CLOUD_INFERENCE", "false").lower() in ("true", "1", "yes")
        self.cloud_inference = cloud_inference if cloud_inference is not None else env_cloud_inf


        self.dense_model_name = dense_model_name or os.getenv(
            "EMBEDDING_MODEL_NAME", "sentence-transformers/all-minilm-l6-v2"
        )
        self.sparse_model_name = sparse_model_name or os.getenv(
            "SPARSE_MODEL_NAME", "qdrant/bm25"
        )
        self.embedding_dim = embedding_dim or int(os.getenv("EMBEDDING_DIM", "384"))

        if not self.cloud_inference:
            self.embeddings = embeddings or SentenceTransformerEmbeddings(
                model_name=self.dense_model_name
            )
        else:
            self.embeddings = None

        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
                if self.url:
                    kwargs: Dict[str, Any] = {"url": self.url, "api_key": self.api_key}
                    if self.cloud_inference:
                        kwargs["cloud_inference"] = True
                    self._client = QdrantClient(**kwargs)
                elif self.location:
                    self._client = QdrantClient(path=self.location)
                else:
                    self._client = QdrantClient(location=":memory:")
            except ImportError:
                raise ImportError(
                    "qdrant-client is not installed. Please install it via `pip install qdrant-client`"
                )
        return self._client

    def ensure_collection(self, recreate: bool = False):
        from qdrant_client.http import models

        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if exists and recreate:
            self.client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            dim = self.embedding_dim if self.embeddings is None else self.embeddings.dimension
            
            # Setup named dense vector and sparse bm25 vector
            vectors_config = {
                "dense": models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                )
            }
            sparse_vectors_config = {
                "bm25": models.SparseVectorParams(
                    index=models.SparseIndexParams(
                        on_disk=False
                    )
                )
            }
            
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=vectors_config,
                    sparse_vectors_config=sparse_vectors_config,
                )
            except Exception:
                # Fallback to standard vector config if sparse config not supported in current environment
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=dim,
                        distance=models.Distance.COSINE,
                    ),
                )

    def add_chunks(self, chunks: List[Chunk], batch_size: int = 64) -> int:
        from qdrant_client.http import models
        from qdrant_client.http.models import PointStruct

        if not chunks:
            return 0

        self.ensure_collection(recreate=False)
        total_indexed = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            points = []

            if self.cloud_inference:
                from qdrant_client.http.models import Document
                for chunk in batch:
                    point_id = abs(hash(chunk.chunk_id)) % (2**63 - 1)
                    payload = {
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "title": chunk.title,
                        "text": chunk.text,
                        "source": chunk.source,
                        "section": chunk.section,
                        "page": chunk.page,
                        "metadata": chunk.metadata,
                    }
                    points.append(
                        PointStruct(
                            id=point_id,
                            payload=payload,
                            vector={
                                "dense": Document(
                                    text=chunk.text,
                                    model=self.dense_model_name,
                                ),
                                "bm25": Document(
                                    text=chunk.text,
                                    model=self.sparse_model_name,
                                ),
                            },
                        )
                    )
            else:
                texts = [c.text for c in batch]
                vectors = self.embeddings.embed_documents(texts)
                for j, chunk in enumerate(batch):
                    point_id = abs(hash(chunk.chunk_id)) % (2**63 - 1)
                    payload = {
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "title": chunk.title,
                        "text": chunk.text,
                        "source": chunk.source,
                        "section": chunk.section,
                        "page": chunk.page,
                        "metadata": chunk.metadata,
                    }
                    points.append(
                        PointStruct(
                            id=point_id,
                            payload=payload,
                            vector={"dense": vectors[j]},
                        )
                    )

            self.client.upsert(collection_name=self.collection_name, points=points)
            total_indexed += len(batch)

        return total_indexed

    def search_dense(self, query: str, top_k: int = 10, query_vector: Optional[List[float]] = None) -> List[RetrievedChunk]:
        """Dense semantic search using all-minilm-l6-v2."""
        retrieved: List[RetrievedChunk] = []

        if self.cloud_inference:
            from qdrant_client.http.models import Document
            try:
                res = self.client.query_points(
                    collection_name=self.collection_name,
                    query=Document(
                        text=query,
                        model=self.dense_model_name,
                    ),
                    using="dense",
                    limit=top_k,
                )
            except Exception:
                res = self.client.query_points(
                    collection_name=self.collection_name,
                    query=Document(
                        text=query,
                        model=self.dense_model_name,
                    ),
                    limit=top_k,
                )

            points = res.points if hasattr(res, "points") else res
            for rank, point in enumerate(points, start=1):
                payload = point.payload or {}
                score = getattr(point, "score", 0.0)
                retrieved.append(
                    RetrievedChunk(
                        chunk_id=payload.get("chunk_id", str(point.id)),
                        document_id=payload.get("document_id", "unknown"),
                        title=payload.get("title"),
                        text=payload.get("text", ""),
                        score=float(score) if score is not None else 0.0,
                        rank=rank,
                        source_method="dense",
                        metadata=payload.get("metadata", {}),
                    )
                )
        else:
            vec = query_vector if query_vector is not None else self.embeddings.embed_query(query)
            try:
                res = self.client.query_points(
                    collection_name=self.collection_name,
                    query=vec,
                    using="dense",
                    limit=top_k,
                )
            except Exception:
                res = self.client.query_points(
                    collection_name=self.collection_name,
                    query=vec,
                    limit=top_k,
                )
            points = res.points if hasattr(res, "points") else res
            for rank, point in enumerate(points, start=1):
                payload = point.payload or {}
                score = getattr(point, "score", 0.0)
                retrieved.append(
                    RetrievedChunk(
                        chunk_id=payload.get("chunk_id", str(point.id)),
                        document_id=payload.get("document_id", "unknown"),
                        title=payload.get("title"),
                        text=payload.get("text", ""),
                        score=float(score) if score is not None else 0.0,
                        rank=rank,
                        source_method="dense",
                        metadata=payload.get("metadata", {}),
                    )
                )

        return retrieved


    def search_bm25(self, query: str, top_k: int = 10) -> List[RetrievedChunk]:
        """Sparse lexical BM25 search using Qdrant cloud inference (qdrant/bm25)."""
        retrieved: List[RetrievedChunk] = []

        if self.cloud_inference:
            from qdrant_client.http.models import Document
            try:
                res = self.client.query_points(
                    collection_name=self.collection_name,
                    query=Document(
                        text=query,
                        model=self.sparse_model_name,
                    ),
                    using="bm25",
                    limit=top_k,
                )
            except Exception:
                # If named 'bm25' vector not present, try 'text' or default query
                res = self.client.query_points(
                    collection_name=self.collection_name,
                    query=Document(
                        text=query,
                        model=self.sparse_model_name,
                    ),
                    using="text",
                    limit=top_k,
                )

            points = res.points if hasattr(res, "points") else res
            for rank, point in enumerate(points, start=1):
                payload = point.payload or {}
                score = getattr(point, "score", 0.0)
                retrieved.append(
                    RetrievedChunk(
                        chunk_id=payload.get("chunk_id", str(point.id)),
                        document_id=payload.get("document_id", "unknown"),
                        title=payload.get("title"),
                        text=payload.get("text", ""),
                        score=float(score) if score is not None else 0.0,
                        rank=rank,
                        source_method="bm25",
                        metadata=payload.get("metadata", {}),
                    )
                )

        return retrieved

    # Default search method maps to search_dense
    def search(self, query: str, top_k: int = 10) -> List[RetrievedChunk]:
        return self.search_dense(query=query, top_k=top_k)
