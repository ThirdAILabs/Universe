from abc import ABC, abstractmethod
from typing import Iterable, List, Tuple

from core.types import ChunkBatch, ChunkId, SupervisedBatch

Score = float


class Retriever(ABC):
    @abstractmethod
    def search(
        self, queries: List[str], top_k: int, **kwargs
    ) -> List[List[Tuple[ChunkId, Score]]]:
        pass

    @abstractmethod
    def rank(
        self, queries: List[str], choices: List[List[ChunkId]], **kwargs
    ) -> List[List[Tuple[ChunkId, Score]]]:
        """For constrained search.
        Note on method signature:
        Choices are provided as a separate argument from queries. While it may
        be safer for the function to accept pairs of (query, choices), choices
        are likely the return value of some function fn(queries) -> choices.
        Thus, there likely exist separate collections for queries and
        choices in memory. This function signature preempts the need to reshape
        these existing data structures.
        """
        pass

    @abstractmethod
    def upvote(self, queries: List[str], chunk_ids: List[ChunkId], **kwargs):
        pass

    @abstractmethod
    def downvote(self, queries: List[str], chunk_ids: List[ChunkId], **kwargs):
        pass

    @abstractmethod
    def associate(self, sources: List[str], targets: List[str], **kwargs):
        pass

    @abstractmethod
    def insert(self, chunks: Iterable[ChunkBatch], **kwargs):
        pass

    @abstractmethod
    def supervised_train(self, samples: Iterable[SupervisedBatch], **kwargs):
        pass

    @abstractmethod
    def delete(self, chunk_ids: List[ChunkId], **kwargs):
        pass
