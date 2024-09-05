from abc import ABC, abstractmethod
from typing import List, Tuple

from .types import Chunk, Score


class Reranker(ABC):
    @abstractmethod
    def rerank(
        self, query: str, results: List[Tuple[Chunk, Score]]
    ) -> List[Tuple[Chunk, Score]]:
        raise NotImplementedError
