from pathlib import Path
from typing import List, Union, Optional
from ..ingestion.pipeline import IngestionPipeline
from ..vectorstore.qdrant_store import QdrantVectorStore
from ..retrieval.bm25_retriever import BM25Retriever
from ..retrieval.dense_retriever import DenseRetriever
from ..retrieval.hybrid_retriever import HybridRetriever
from ..retrieval.reranker import FlashRankReranker, BGEReranker
from ..generation.answer_generator import AnswerGenerator
from .schemas import Chunk, RAGResponse
from .pipeline import RAGPipeline


class RAGService:
    """Service layer managing document indexing and Hybrid RAG query execution."""

    def __init__(self):
        self.ingestion = IngestionPipeline()
        self.vectorstore = QdrantVectorStore()
        self.dense_retriever = DenseRetriever(vectorstore=self.vectorstore)
        self.bm25_retriever = BM25Retriever(vectorstore=self.vectorstore)
        self.hybrid_retriever = HybridRetriever(
            dense_retriever=self.dense_retriever,
            bm25_retriever=self.bm25_retriever,
        )
        self.reranker = FlashRankReranker()
        self.answer_generator = AnswerGenerator()

        self.pipeline = RAGPipeline(
            dense_retriever=self.dense_retriever,
            bm25_retriever=self.bm25_retriever,
            hybrid_retriever=self.hybrid_retriever,
            reranker=self.reranker,
            answer_generator=self.answer_generator,
        )

    def ingest_directory(self, directory_path: Union[str, Path] = "data", limit_per_file: Optional[int] = None) -> int:
        """Ingests all files from a directory (e.g. data/) and indexes into Qdrant & BM25."""
        chunks: List[Chunk] = self.ingestion.process_directory(
            directory_path=directory_path,
            limit_per_file=limit_per_file,
        )
        total_indexed = self.vectorstore.add_chunks(chunks)
        self.bm25_retriever.index_chunks(chunks)
        return total_indexed

    def ingest_and_index(self, file_path: Union[str, Path], limit: Optional[int] = None) -> int:
        file_path = Path(file_path)
        if file_path.is_dir():
            return self.ingest_directory(directory_path=file_path, limit_per_file=limit)

        chunks: List[Chunk] = self.ingestion.process_file(file_path, limit=limit)
        total_indexed = self.vectorstore.add_chunks(chunks)
        self.bm25_retriever.index_chunks(chunks)
        return total_indexed

    def warmup(self):
        """Pre-warms and pre-loads embedding and reranker models into RAM for zero-latency first query."""
        try:
            if self.vectorstore and self.vectorstore.embeddings:
                _ = self.vectorstore.embeddings.model
        except Exception:
            pass

        try:
            if self.reranker:
                _ = self.reranker.model
        except Exception:
            pass

    def query(self, query: str, top_k: int = 5) -> RAGResponse:
        """Executes Hybrid RAG query directly."""
        return self.pipeline.query(query=query, top_k=top_k)

    def query_stream(self, query: str, top_k: int = 5):
        """Executes streaming Hybrid RAG query."""
        return self.pipeline.query_stream(query=query, top_k=top_k)

