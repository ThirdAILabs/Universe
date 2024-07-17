from abc import ABC, abstractmethod
from enum import Enum
from typing import Iterable, List, Set

from pandas.api.types import is_numeric_dtype, is_string_dtype

from .types import (
    Chunk,
    ChunkBatch,
    ChunkId,
    CustomIdSupervisedBatch,
    SupervisedBatch,
    VersionedNewChunkBatch,
)


class CustomIDType(Enum):
    NotSet = 1
    NoneType = 2
    String = 3
    Integer = 4


# Calling this ChunkStore instead of DocumentStore because it stores chunks
# instead of documents.
class ChunkStore(ABC):
    def __init__(self):
        self.custom_id_type = CustomIDType.NotSet

    @abstractmethod
    def insert(
        self, chunks: Iterable[VersionedNewChunkBatch], **kwargs
    ) -> Iterable[ChunkBatch]:
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

    @abstractmethod
    def get_doc_chunks(self, doc_id: str, before_version: int) -> Set[ChunkId]:
        raise NotImplementedError

    @abstractmethod
    def max_version_for_doc(self, doc_id: str) -> int:
        raise NotImplementedError

    def _set_or_validate_custom_id_type(self, custom_ids):
        incoming_custom_id_type = CustomIDType.NotSet
        if custom_ids is None:
            incoming_custom_id_type = CustomIDType.NoneType
        elif is_string_dtype(custom_ids):
            incoming_custom_id_type = CustomIDType.String
        elif is_numeric_dtype(custom_ids):
            incoming_custom_id_type = CustomIDType.Integer
        else:
            raise ValueError("Invalid custom id type.")

        if self.custom_id_type == CustomIDType.NotSet:
            self.custom_id_type = incoming_custom_id_type
        elif incoming_custom_id_type != self.custom_id_type:
            raise ValueError(
                "Custom ids must all have the same type. Must be int, str, or None."
            )
