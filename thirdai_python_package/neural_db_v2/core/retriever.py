from abc import ABC, abstractmethod
from typing import List, Tuple, Iterable
from core.types import ChunkId, Chunk


Score = float


class Retriever(ABC):
    @abstractmethod
    def search(self, query: str, top_k: int, **kwargs) -> List[Tuple[ChunkId, Score]]:
        pass

    @abstractmethod
    def search_batch(
        self, queries: List[str], top_k: int, **kwargs
    ) -> List[List[Tuple[ChunkId, Score]]]:
        pass

    @abstractmethod
    def rank(
        self, query: str, choices: List[ChunkId], **kwargs
    ) -> List[Tuple[ChunkId, Score]]:
        """For constrained search."""
        pass

    @abstractmethod
    def rank_batch(
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

    # We should discourage single sample RLHF since it does not work well.
    @abstractmethod
    def upvote_batch(self, queries: List[str], chunk_ids: List[ChunkId], **kwargs):
        pass

    @abstractmethod
    def downvote_batch(self, queries: List[str], chunk_ids: List[ChunkId], **kwargs):
        pass

    @abstractmethod
    def associate_batch(self, sources: List[str], targets: List[str], **kwargs):
        pass

    @abstractmethod
    def dissociate_batch(self, sources: List[str], targets: List[str], **kwargs):
        pass

    @abstractmethod
    def insert_batch(self, chunks: Iterable[Chunk], **kwargs):
        pass

    @abstractmethod
    def delete(self, chunk_id: ChunkId, **kwargs):
        pass

    @abstractmethod
    def delete_batch(self, chunk_ids: List[ChunkId], **kwargs):
        pass
