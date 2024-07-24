from abc import ABC, abstractmethod
from typing import Iterable, List, Set, Tuple

from .documents import Document
from .types import Chunk, ChunkBatch, ChunkId, InsertedDocMetadata


# Calling this ChunkStore instead of DocumentStore because it stores chunks
# instead of documents.
class ChunkStore(ABC):
    @abstractmethod
    def insert(
        self, docs: List[Document], **kwargs
    ) -> Tuple[Iterable[ChunkBatch], List[InsertedDocMetadata]]:
        raise NotImplementedError

    @abstractmethod
    def delete(self, chunk_ids: List[ChunkId], **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_chunks(self, chunk_ids: List[ChunkId], **kwargs) -> List[Chunk]:
        raise NotImplementedError

    @abstractmethod
    def filter_chunk_ids(self, constraints: dict, **kwargs) -> Set[ChunkId]:
        raise NotImplementedError

    @abstractmethod
    def get_doc_chunks(self, doc_id: str, before_version: int) -> List[ChunkId]:
        raise NotImplementedError

    @abstractmethod
    def max_version_for_doc(self, doc_id: str) -> int:
        raise NotImplementedError
