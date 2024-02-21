from abc import ABC, abstractmethod
from typing import Iterable, List, Set, Union

from core.types import (
    Chunk,
    ChunkId,
    NewChunk,
    CustomIdSupervisedSample,
    SupervisedSample,
)


# Calling this ChunkStore instead of DocumentStore because it stores chunks
# instead of documents.
class ChunkStore(ABC):
    @abstractmethod
    def insert(self, chunks: Iterable[NewChunk], **kwargs) -> Iterable[Chunk]:
        pass

    @abstractmethod
    def delete(self, chunk_ids: List[ChunkId], **kwargs):
        pass

    @abstractmethod
    def get_chunks(self, chunk_ids: List[ChunkId], **kwargs) -> List[Chunk]:
        pass

    @abstractmethod
    def filter_chunk_ids(self, constraints: dict, **kwargs) -> Set[ChunkId]:
        pass

    @abstractmethod
    def remap_custom_ids(
        self, samples: Iterable[CustomIdSupervisedSample]
    ) -> Iterable[SupervisedSample]:
        pass
