import concurrent.futures
from typing import List, Optional
from ..rag.schemas import RetrievedChunk
from .dense_retriever import DenseRetriever
from .bm25_retriever import BM25Retriever
from .rrf import ReciprocalRankFusion


class HybridRetriever:
    """Combines Dense Semantic and Sparse BM25 retrieval concurrently using parallel threads and RRF."""

    def __init__(
        self,
        dense_retriever: Optional[DenseRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        rrf_k: int = 60,
        max_workers: int = 2,
    ):
        self.dense_retriever = dense_retriever or DenseRetriever()
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.fusion = ReciprocalRankFusion(k=rrf_k)
        self.max_workers = max_workers

    def retrieve(
        self,
        query: str,
        top_k_dense: int = 10,
        top_k_sparse: int = 10,
        top_k_fused: int = 10,
        query_vector: Optional[List[float]] = None,
    ) -> List[RetrievedChunk]:
        """Runs Dense semantic retrieval and BM25 lexical retrieval concurrently in parallel."""
        dense_results: List[RetrievedChunk] = []
        sparse_results: List[RetrievedChunk] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_dense = executor.submit(
                self.dense_retriever.retrieve,
                query=query,
                top_k=top_k_dense,
                query_vector=query_vector,
            )
            future_sparse = executor.submit(
                self.bm25_retriever.retrieve,
                query=query,
                top_k=top_k_sparse,
            )

            try:
                dense_results = future_dense.result()
            except Exception as e:
                dense_results = []

            try:
                sparse_results = future_sparse.result()
            except Exception as e:
                sparse_results = []

        fused_results = self.fusion.fuse(
            ranked_lists=[dense_results, sparse_results],
            top_k=top_k_fused,
        )
        return fused_results


