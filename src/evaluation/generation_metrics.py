import re
import string
from collections import Counter
from typing import List


class GenerationMetrics:
    """Calculates Exact Match (EM), Token-level F1, and citation presence."""

    @staticmethod
    def normalize_answer(s: str) -> str:
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @classmethod
    def exact_match_score(cls, prediction: str, ground_truth: str) -> float:
        return 1.0 if cls.normalize_answer(prediction) == cls.normalize_answer(ground_truth) else 0.0

    @classmethod
    def f1_score(cls, prediction: str, ground_truth: str) -> float:
        prediction_tokens = cls.normalize_answer(prediction).split()
        ground_truth_tokens = cls.normalize_answer(ground_truth).split()

        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())

        if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
            return 1.0 if prediction_tokens == ground_truth_tokens else 0.0

        if num_same == 0:
            return 0.0

        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
