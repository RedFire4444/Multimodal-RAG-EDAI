from typing import List, Dict, Any
from tqdm import tqdm
from ..rag.schemas import EvaluationSample
from ..rag.pipeline import RAGPipeline
from .retrieval_metrics import RetrievalMetrics
from .generation_metrics import GenerationMetrics
from .ragas_evaluator import RagasEvaluator


class BenchmarkEvaluator:
    """Evaluates the Hybrid RAG system across Retrieval and Generation benchmarks."""

    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline
        self.ragas_evaluator = RagasEvaluator()

    def evaluate_retrieval(
        self,
        samples: List[EvaluationSample],
        top_k: int = 10,
    ) -> Dict[str, float]:
        recalls_1, recalls_3, recalls_5, recalls_10 = [], [], [], []
        mrrs = []
        ndcg_5, ndcg_10 = [], []

        for sample in tqdm(samples, desc="Evaluating Hybrid Retrieval"):
            retrieved = self.pipeline.retrieve(query=sample.question, top_k=top_k)
            retrieved_doc_titles = [r.title or r.document_id for r in retrieved]

            recalls_1.append(RetrievalMetrics.recall_at_k(retrieved_doc_titles, sample.gold_documents, k=1))
            recalls_3.append(RetrievalMetrics.recall_at_k(retrieved_doc_titles, sample.gold_documents, k=3))
            recalls_5.append(RetrievalMetrics.recall_at_k(retrieved_doc_titles, sample.gold_documents, k=5))
            recalls_10.append(RetrievalMetrics.recall_at_k(retrieved_doc_titles, sample.gold_documents, k=10))
            mrrs.append(RetrievalMetrics.mrr(retrieved_doc_titles, sample.gold_documents))
            ndcg_5.append(RetrievalMetrics.ndcg_at_k(retrieved_doc_titles, sample.gold_documents, k=5))
            ndcg_10.append(RetrievalMetrics.ndcg_at_k(retrieved_doc_titles, sample.gold_documents, k=10))

        return {
            "mode": "Hybrid RAG",
            "samples_count": len(samples),
            "Recall@1": sum(recalls_1) / len(recalls_1) if recalls_1 else 0.0,
            "Recall@3": sum(recalls_3) / len(recalls_3) if recalls_3 else 0.0,
            "Recall@5": sum(recalls_5) / len(recalls_5) if recalls_5 else 0.0,
            "Recall@10": sum(recalls_10) / len(recalls_10) if recalls_10 else 0.0,
            "MRR": sum(mrrs) / len(mrrs) if mrrs else 0.0,
            "nDCG@5": sum(ndcg_5) / len(ndcg_5) if ndcg_5 else 0.0,
            "nDCG@10": sum(ndcg_10) / len(ndcg_10) if ndcg_10 else 0.0,
        }

    def evaluate_generation(
        self,
        samples: List[EvaluationSample],
        top_k: int = 5,
        run_ragas: bool = False,
    ) -> Dict[str, Any]:
        em_scores, f1_scores = [], []
        questions, answers, contexts, ground_truths = [], [], [], []

        for sample in tqdm(samples, desc="Evaluating Hybrid Generation"):
            response = self.pipeline.query(query=sample.question, top_k=top_k)
            em = GenerationMetrics.exact_match_score(response.answer, sample.gold_answer)
            f1 = GenerationMetrics.f1_score(response.answer, sample.gold_answer)

            em_scores.append(em)
            f1_scores.append(f1)

            if run_ragas:
                questions.append(sample.question)
                answers.append(response.answer)
                contexts.append([c.text for c in response.evidence_chunks])
                ground_truths.append(sample.gold_answer)

        results = {
            "mode": "Hybrid RAG",
            "Exact_Match": sum(em_scores) / len(em_scores) if em_scores else 0.0,
            "F1_Score": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        }

        if run_ragas and questions:
            ragas_metrics = self.ragas_evaluator.evaluate(
                questions=questions,
                answers=answers,
                contexts=contexts,
                ground_truths=ground_truths,
            )
            results.update(ragas_metrics)

        return results
