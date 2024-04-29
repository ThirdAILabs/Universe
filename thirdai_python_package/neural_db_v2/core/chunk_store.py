from abc import ABC, abstractmethod
from typing import Iterable, List, Set

from .types import (
    Chunk,
    ChunkBatch,
    ChunkId,
    CustomIdSupervisedBatch,
    NewChunkBatch,
    SupervisedBatch,
)


# Calling this ChunkStore instead of DocumentStore because it stores chunks
# instead of documents.
class ChunkStore(ABC):
    @abstractmethod
    def insert(self, chunks: Iterable[NewChunkBatch], **kwargs) -> Iterable[ChunkBatch]:
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
    def remap_custom_ids(
        self, samples: Iterable[CustomIdSupervisedBatch]
    ) -> Iterable[SupervisedBatch]:
        raise NotImplementedError
