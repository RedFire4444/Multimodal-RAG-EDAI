from typing import List, Dict, Any


class RagasEvaluator:
    """Wrapper for RAGAS evaluation (Faithfulness, Answer Relevancy, Context Precision, Context Recall)."""

    def __init__(self):
        pass

    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str],
    ) -> Dict[str, float]:
        """Runs RAGAS evaluation pipeline."""
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                answer_correctness,
            )

            data = {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
            dataset = Dataset.from_dict(data)

            results = evaluate(
                dataset=dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                    answer_correctness,
                ],
            )
            return dict(results)
        except Exception as e:
            return {"ragas_error": str(e)}
