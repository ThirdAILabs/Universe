from abc import ABC, abstractmethod
from typing import Iterable, List, Set, Tuple

from .types import ChunkBatch, ChunkId, Score, SupervisedBatch


class Retriever(ABC):
    @abstractmethod
    def search(
        self, queries: List[str], top_k: int, **kwargs
    ) -> List[List[Tuple[ChunkId, Score]]]:
        raise NotImplementedError

    @abstractmethod
    def rank(
        self, queries: List[str], choices: List[Set[ChunkId]], **kwargs
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
        raise NotImplementedError

    @abstractmethod
    def upvote(self, queries: List[str], chunk_ids: List[ChunkId], **kwargs):
        raise NotImplementedError

    @abstractmethod
    def associate(self, sources: List[str], targets: List[str], **kwargs):
        raise NotImplementedError

    @abstractmethod
    def insert(self, chunks: Iterable[ChunkBatch], **kwargs):
        raise NotImplementedError

    @abstractmethod
    def supervised_train(self, samples: Iterable[SupervisedBatch], **kwargs):
        raise NotImplementedError

    @abstractmethod
    def delete(self, chunk_ids: List[ChunkId], **kwargs):
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(path: str):
        raise NotImplementedError
