from typing import List, Optional, Dict, Any
import os
import math
from ..rag.schemas import RetrievedChunk


class FlashRankReranker:
    """Ultra-fast, lightweight ONNX-based reranker using FlashRank.
    Runs locally on CPU with zero PyTorch/GPU overhead.
    Implements class-level caching so the ranker model is loaded only once.
    """

    _CACHE: Dict[str, Any] = {}

    def __init__(
        self,
        model_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        # Default model is ms-marco-TinyBERT-L-2-v2 (ultra-fast, ~4MB)
        # Other supported models: ms-marco-MiniLM-L-12-v2, rank-T5-flan, etc.
        self.model_name = model_name or os.getenv(
            "RERANKER_MODEL_NAME", "ms-marco-TinyBERT-L-2-v2"
        )
        self.cache_dir = cache_dir

    @property
    def model(self):
        cache_key = (self.model_name, self.cache_dir)
        if cache_key not in self._CACHE:
            try:
                from flashrank import Ranker
                kwargs = {"model_name": self.model_name}
                if self.cache_dir:
                    kwargs["cache_dir"] = self.cache_dir
                self._CACHE[cache_key] = Ranker(**kwargs)
            except ImportError:
                # Fallback to langchain compressor if flashrank directly not found
                try:
                    from langchain_community.document_compressors import FlashrankRerank
                    self._CACHE[cache_key] = FlashrankRerank(model=self.model_name)
                except ImportError as err:
                    raise ImportError(
                        "FlashRank is not installed. Please run: pip install flashrank"
                    ) from err

        return self._CACHE[cache_key]

    def compute_score(self, pairs: List[List[str]]) -> List[float]:
        """Computes reranking relevance score for [query, passage] pairs."""
        if not pairs:
            return []

        # If all pairs share the same query, run single RerankRequest
        first_query = pairs[0][0]
        all_same_query = all(p[0] == first_query for p in pairs)

        try:
            from flashrank import RerankRequest

            if all_same_query:
                passages = [{"id": i, "text": p[1]} for i, p in enumerate(pairs)]
                req = RerankRequest(query=first_query, passages=passages)
                results = self.model.rerank(req)
                # Map scores back to original order
                score_map = {res["id"]: float(res.get("score", 0.0)) for res in results}
                return [score_map.get(i, 0.0) for i in range(len(pairs))]
            else:
                scores = []
                for q, passage_text in pairs:
                    req = RerankRequest(query=q, passages=[{"id": 0, "text": passage_text}])
                    res = self.model.rerank(req)
                    scores.append(float(res[0].get("score", 0.0)) if res else 0.0)
                return scores
        except Exception:
            return [0.0] * len(pairs)

    def rerank(
        self, query: str, candidate_chunks: List[RetrievedChunk], top_n: int = 5
    ) -> List[RetrievedChunk]:
        """Reranks candidate chunks based on FlashRank cross-encoder scores."""
        if not candidate_chunks:
            return []

        try:
            from flashrank import RerankRequest

            passages = [
                {
                    "id": i,
                    "text": chunk.text,
                    "chunk": chunk,
                }
                for i, chunk in enumerate(candidate_chunks)
            ]

            req = RerankRequest(query=query, passages=passages)
            ranked_results = self.model.rerank(req)

            reranked: List[RetrievedChunk] = []
            for new_rank, res in enumerate(ranked_results[:top_n], start=1):
                chunk: RetrievedChunk = res.get("chunk") or candidate_chunks[res["id"]]
                reranked.append(
                    RetrievedChunk(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        title=chunk.title,
                        text=chunk.text,
                        score=float(res.get("score", 0.0)),
                        rank=new_rank,
                        source_method="reranker",
                        metadata=chunk.metadata,
                    )
                )

            return reranked

        except Exception as e:
            # Fallback to candidates with default rank if reranker encounters issue
            return candidate_chunks[:top_n]


# Alias for backward compatibility
BGEReranker = FlashRankReranker
