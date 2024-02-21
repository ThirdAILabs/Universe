from abc import ABC, abstractmethod
from typing import Iterable, List, Set, Union

from core.types import Chunk, ChunkId, NewChunk


# Calling this ChunkStore instead of DocumentStore because it stores chunks
# instead of documents.
class ChunkStore(ABC):
    @abstractmethod
    def insert(self, chunk: NewChunk, **kwargs) -> Chunk:
        pass

    @abstractmethod
    def insert_batch(self, chunks: Iterable[NewChunk], **kwargs) -> Iterable[Chunk]:
        pass

    @abstractmethod
    def delete(self, chunk_id: ChunkId, **kwargs):
        pass

    @abstractmethod
    def delete_batch(self, chunk_ids: List[ChunkId], **kwargs):
        pass

    @abstractmethod
    def get_chunk(self, chunk_id: ChunkId, **kwargs) -> Chunk:
        pass

    @abstractmethod
    def get_chunk_batch(self, chunk_ids: List[ChunkId], **kwargs) -> List[Chunk]:
        pass

    @abstractmethod
    def matching_chunk_ids(self, constraints: dict, **kwargs) -> Set[ChunkId]:
        pass

    @abstractmethod
    def user_assigned_ids_to_chunk_ids(
        self, user_assigned_ids: List[Union[str, int]]
    ) -> List[ChunkId]:
        pass
