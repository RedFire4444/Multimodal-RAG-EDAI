import json
from pathlib import Path
from typing import List, Union
from ..rag.schemas import EvaluationSample


class GoldenDatasetLoader:
    """Loads and samples curated evaluation benchmarks from 2WikiMultiHopQA datasets."""

    @staticmethod
    def load_golden_set(file_path: Union[str, Path], limit: int = 1000) -> List[EvaluationSample]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Golden dataset file not found: {file_path}")

        samples: List[EvaluationSample] = []
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        items = data[:limit] if limit else data
        for idx, item in enumerate(items):
            # Extract supporting facts/documents
            # 2WikiMultiHopQA format: 'supporting_facts': [[title, sentence_index], ...]
            supporting_facts = item.get("supporting_facts", [])
            gold_docs = list(set([fact[0] for fact in supporting_facts if isinstance(fact, (list, tuple))]))

            samples.append(
                EvaluationSample(
                    sample_id=item.get("_id", f"sample_{idx}"),
                    question=item.get("question", ""),
                    gold_answer=str(item.get("answer", "")),
                    gold_documents=gold_docs,
                    gold_passages=[f"{fact[0]}_{fact[1]}" for fact in supporting_facts if len(fact) > 1],
                    question_type=item.get("type", "multi-hop"),
                )
            )

        return samples
