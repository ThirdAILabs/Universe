from abc import ABC, abstractmethod
from typing import List, Iterable, Set
from pathlib import Path
from core.types import ChunkId, NewChunk, Chunk


class ChunkStore(ABC):
    @abstractmethod
    def insert(self, chunk: NewChunk, **kwargs) -> Chunk:
        pass

    @abstractmethod
    def insert_batch(self, chunks: Iterable[NewChunk], **kwargs) -> Iterable[Chunk]:
        pass

    @abstractmethod
    def delete(self, doc_id: ChunkId, **kwargs):
        pass

    @abstractmethod
    def delete_batch(self, doc_ids: List[ChunkId], **kwargs):
        pass

    @abstractmethod
    def get_chunk(self, doc_id: ChunkId, **kwargs) -> Chunk:
        pass

    @abstractmethod
    def get_chunk_batch(self, doc_ids: List[ChunkId], **kwargs) -> List[Chunk]:
        pass

    @abstractmethod
    def matching_doc_ids(self, constraints: dict, **kwargs) -> Set[ChunkId]:
        pass
