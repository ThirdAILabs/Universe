from abc import ABC, abstractmethod
from typing import Iterable, List, Set, Tuple

from .types import Chunk, ChunkBatch, ChunkId, InsertedDocMetadata, NewChunkBatch


# Calling this ChunkStore instead of DocumentStore because it stores chunks
# instead of documents.
class ChunkStore(ABC):
    @abstractmethod
    def insert(
        self, chunks: Iterable[Iterable[NewChunkBatch]], **kwargs
    ) -> Tuple[Iterable[ChunkBatch], Iterable[InsertedDocMetadata]]:
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
