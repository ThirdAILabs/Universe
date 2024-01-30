from abc import ABC, abstractmethod
from typing import List, Tuple, Iterable
from pathlib import Path
from core.types import DocId, Document


Score = float


class Retriever(ABC):
    @abstractmethod
    def find(self, query: str, top_k: int, **kwargs) -> List[Tuple[DocId, Score]]:
        pass

    @abstractmethod
    def find_batch(
        self, queries: List[str], top_k: int, **kwargs
    ) -> List[List[Tuple[DocId, Score]]]:
        pass

    @abstractmethod
    def rank(
        self, query: str, choices: List[DocId], **kwargs
    ) -> List[Tuple[DocId, Score]]:
        pass

    @abstractmethod
    def rank_batch(
        self, queries: List[str], choices: List[List[DocId]], **kwargs
    ) -> List[List[Tuple[DocId, Score]]]:
        """
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
    def upvote_batch(self, queries: List[str], doc_ids: List[DocId], **kwargs):
        pass

    @abstractmethod
    def downvote_batch(self, queries: List[str], doc_ids: List[DocId], **kwargs):
        pass

    @abstractmethod
    def associate_batch(self, sources: List[str], targets: List[str], **kwargs):
        pass

    @abstractmethod
    def dissociate_batch(self, sources: List[str], targets: List[str], **kwargs):
        pass

    @abstractmethod
    def insert_batch(self, docs: Iterable[Document], checkpoint: Path, **kwargs):
        pass

    @abstractmethod
    def delete(self, doc_id: DocId, **kwargs):
        pass

    @abstractmethod
    def delete_batch(self, doc_ids: List[DocId], **kwargs):
        pass
