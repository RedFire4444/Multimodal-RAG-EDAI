import os
import time
from typing import List, Optional, Dict, Any

from .schemas import RetrievedChunk, RAGResponse
from ..retrieval.dense_retriever import DenseRetriever
from ..retrieval.bm25_retriever import BM25Retriever
from ..retrieval.hybrid_retriever import HybridRetriever
from ..retrieval.reranker import FlashRankReranker, BGEReranker
from ..generation.answer_generator import AnswerGenerator


class RAGPipeline:
    """Hybrid RAG Pipeline:
    Combines Dense Semantic Search (SentenceTransformer + Qdrant) and
    Sparse Lexical Search (BM25) fused with Reciprocal Rank Fusion (RRF)
    and refined with FlashRank Reranking,
    followed by Grounded Answer Generation with Groq LLM.
    """

    def __init__(
        self,
        dense_retriever: Optional[DenseRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        hybrid_retriever: Optional[HybridRetriever] = None,
        reranker: Optional[FlashRankReranker] = None,
        answer_generator: Optional[AnswerGenerator] = None,
    ):
        self.dense_retriever = dense_retriever or DenseRetriever()
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.hybrid_retriever = (
            hybrid_retriever
            or HybridRetriever(
                dense_retriever=self.dense_retriever,
                bm25_retriever=self.bm25_retriever,
            )
        )
        self.reranker = reranker or FlashRankReranker()
        self.answer_generator = answer_generator or AnswerGenerator()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        top_k_dense: Optional[int] = None,
        top_k_sparse: Optional[int] = None,
        top_k_fused: Optional[int] = None,
        query_vector: Optional[List[float]] = None,
    ) -> List[RetrievedChunk]:
        """Performs Hybrid retrieval (Dense Top-20 + BM25 Top-20 -> RRF Top 15-20 -> Reranker Top 5)."""
        dense_k = top_k_dense or int(os.getenv("TOP_K_DENSE", "20"))
        sparse_k = top_k_sparse or int(os.getenv("TOP_K_SPARSE", "20"))
        fused_k = top_k_fused or int(os.getenv("TOP_K_FUSED", "15"))

        # Retrieve candidate pool from Dense + BM25 via parallel RRF (Top 15-20)
        candidate_pool = self.hybrid_retriever.retrieve(
            query=query,
            top_k_dense=dense_k,
            top_k_sparse=sparse_k,
            top_k_fused=fused_k,
            query_vector=query_vector,
        )

        # Apply Cross-Encoder Reranker on reduced candidate pool (Top 15-20 -> Top 5)
        try:
            reranked_chunks = self.reranker.rerank(
                query=query, candidate_chunks=candidate_pool, top_n=top_k
            )
            return reranked_chunks
        except Exception:
            return candidate_pool[:top_k]


    def query(self, query: str, top_k: int = 5) -> RAGResponse:
        """Executes full Hybrid RAG pipeline: Retrieval -> Grounded Generation."""
        start_time = time.time()
        evidence_chunks = self.retrieve(query=query, top_k=top_k)
        retrieval_latency = time.time() - start_time

        gen_start = time.time()
        answer = self.answer_generator.generate_answer(
            query=query, evidence_chunks=evidence_chunks
        )
        gen_latency = time.time() - gen_start
        total_latency = time.time() - start_time

        return RAGResponse(
            query=query,
            answer=answer,
            evidence_chunks=evidence_chunks,
            mode="hybrid",
            metadata={
                "retrieval_latency_sec": round(retrieval_latency, 4),
                "generation_latency_sec": round(gen_latency, 4),
                "total_latency_sec": round(total_latency, 4),
                "num_evidence_chunks": len(evidence_chunks),
            },
        )

    def query_stream(self, query: str, top_k: int = 5):
        """Streams the Hybrid RAG generation tokens as they arrive while calculating metrics."""
        start_time = time.time()
        evidence_chunks = self.retrieve(query=query, top_k=top_k)
        retrieval_latency = time.time() - start_time

        yield ("evidence", evidence_chunks, retrieval_latency)

        gen_start = time.time()
        answer_parts = []
        for token in self.answer_generator.generate_answer_stream(query=query, evidence_chunks=evidence_chunks):
            answer_parts.append(token)
            yield ("token", token, None)

        gen_latency = time.time() - gen_start
        total_latency = time.time() - start_time
        full_answer = "".join(answer_parts).strip()

        response = RAGResponse(
            query=query,
            answer=full_answer,
            evidence_chunks=evidence_chunks,
            mode="hybrid",
            metadata={
                "retrieval_latency_sec": round(retrieval_latency, 4),
                "generation_latency_sec": round(gen_latency, 4),
                "total_latency_sec": round(total_latency, 4),
                "num_evidence_chunks": len(evidence_chunks),
            },
        )

        yield ("done", full_answer, response)

