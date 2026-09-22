import math
from typing import List, Set


class RetrievalMetrics:
    """Calculates classical Information Retrieval (IR) metrics:
    - Recall@K
    - Precision@K
    - Mean Reciprocal Rank (MRR)
    - Normalized Discounted Cumulative Gain (nDCG@K)
    """

    @staticmethod
    def recall_at_k(retrieved_docs: List[str], gold_docs: List[str], k: int) -> float:
        if not gold_docs:
            return 1.0
        retrieved_k = set(retrieved_docs[:k])
        gold_set = set(gold_docs)
        intersection = retrieved_k.intersection(gold_set)
        return len(intersection) / len(gold_set)

    @staticmethod
    def precision_at_k(retrieved_docs: List[str], gold_docs: List[str], k: int) -> float:
        if k <= 0:
            return 0.0
        retrieved_k = retrieved_docs[:k]
        gold_set = set(gold_docs)
        intersection = [doc for doc in retrieved_k if doc in gold_set]
        return len(intersection) / k

    @staticmethod
    def mrr(retrieved_docs: List[str], gold_docs: List[str]) -> float:
        gold_set = set(gold_docs)
        for rank, doc in enumerate(retrieved_docs, start=1):
            if doc in gold_set:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved_docs: List[str], gold_docs: List[str], k: int) -> float:
        gold_set = set(gold_docs)
        if not gold_set or k <= 0:
            return 0.0

        dcg = 0.0
        for i, doc in enumerate(retrieved_docs[:k]):
            rel = 1.0 if doc in gold_set else 0.0
            dcg += (2**rel - 1) / math.log2(i + 2)

        # Ideal DCG
        ideal_rels = [1.0] * min(len(gold_set), k) + [0.0] * max(0, k - len(gold_set))
        idcg = sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal_rels))

        return dcg / idcg if idcg > 0 else 0.0
