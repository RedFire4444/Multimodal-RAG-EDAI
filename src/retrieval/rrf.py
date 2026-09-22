from typing import List, Dict
from collections import defaultdict
from ..rag.schemas import RetrievedChunk


class ReciprocalRankFusion:
    """Combines multiple ranked retrieval lists using Reciprocal Rank Fusion (RRF).

    Formula: RRF_Score(d) = Sum_{m} (1 / (k + rank_m(d)))
    """

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(
        self, ranked_lists: List[List[RetrievedChunk]], top_k: int = 10
    ) -> List[RetrievedChunk]:
        scores: Dict[str, float] = defaultdict(float)
        chunk_map: Dict[str, RetrievedChunk] = {}

        for ranked_list in ranked_lists:
            for rank, chunk in enumerate(ranked_list, start=1):
                chunk_id = chunk.chunk_id
                scores[chunk_id] += 1.0 / (self.k + rank)
                if chunk_id not in chunk_map:
                    chunk_map[chunk_id] = chunk

        sorted_chunks = sorted(
            scores.items(), key=lambda item: item[1], reverse=True
        )[:top_k]

        fused_results: List[RetrievedChunk] = []
        for new_rank, (chunk_id, score) in enumerate(sorted_chunks, start=1):
            original = chunk_map[chunk_id]
            fused_results.append(
                RetrievedChunk(
                    chunk_id=original.chunk_id,
                    document_id=original.document_id,
                    title=original.title,
                    text=original.text,
                    score=score,
                    rank=new_rank,
                    source_method="rrf",
                    metadata=original.metadata,
                )
            )

        return fused_results
