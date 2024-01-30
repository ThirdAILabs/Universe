from abc import ABC, abstractmethod
from typing import List, Iterable, Set
from pathlib import Path
from core.types import DocId, Document


class Index(ABC):
    @abstractmethod
    def insert_batch(
        self,
        docs: Iterable[Document],
        assign_new_unique_ids: bool = True,
        checkpoint: Path = None,
        **kwargs,
    ):
        pass

    @abstractmethod
    def delete(self, doc_id: DocId, **kwargs):
        pass

    @abstractmethod
    def delete_batch(self, doc_ids: List[DocId], **kwargs):
        pass

    @abstractmethod
    def get_doc(self, doc_id: DocId, **kwargs):
        pass

    @abstractmethod
    def get_doc_batch(self, doc_ids: List[DocId], **kwargs):
        pass

    @abstractmethod
    def matching_doc_ids(self, constraints: dict, **kwargs) -> Set[DocId]:
        pass
