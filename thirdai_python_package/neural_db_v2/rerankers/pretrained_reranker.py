from typing import List, Tuple

from transformers import AutoModelForSequenceClassification

from ..core.reranker import Reranker
from ..core.types import Chunk, Score


class PretrainedReranker(Reranker):
    def __init__(self):
        super().__init__()

        self.model = AutoModelForSequenceClassification.from_pretrained(
            "jinaai/jina-reranker-v1-tiny-en",
            num_labels=1,
            trust_remote_code=True,
            max_position_embeddings=4096,
        )

    def rerank(
        self, query: str, results: List[Tuple[Chunk, Score]]
    ) -> List[Tuple[Chunk, Score]]:
        if len(results) < 2:
            return results
        new_scores = self.model.compute_score(
            [(query, chunk.keywords + " " + chunk.text) for chunk, _ in results]
        )

        results = [
            (chunk, new_score) for (chunk, _), new_score in zip(results, new_scores)
        ]
        results.sort(key=lambda x: x[1], reverse=True)

        return results
