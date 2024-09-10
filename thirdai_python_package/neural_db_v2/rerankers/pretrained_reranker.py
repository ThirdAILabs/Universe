from typing import List, Tuple

from transformers import AutoModelForSequenceClassification

from ..core.reranker import Reranker
from ..core.types import Chunk, Score


class PretrainedReranker(Reranker):
    def __init__(self):
        super().__init__()

        # Testing code using this reranker causes mac wheel builds to time out.

        self.model = AutoModelForSequenceClassification.from_pretrained(
            "jinaai/jina-reranker-v1-tiny-en", num_labels=1, trust_remote_code=True
        )

    def rerank(
        self, query: str, results: List[Tuple[Chunk, Score]]
    ) -> List[Tuple[Chunk, Score]]:
        new_scores = self.model.compute_score(
            [(query, chunk.keywords + " " + chunk.text) for chunk, _ in results]
        )

        results = [
            (chunk, new_score) for (chunk, _), new_score in zip(results, new_scores)
        ]
        results.sort(key=lambda x: x[1], reverse=True)

        return results
