import random
from abc import ABC, abstractmethod
from typing import Iterable, List, Set, Tuple

import pandas as pd

from ..core.types import (
    ChunkMetaDataSummary,
    MetadataType,
    NumericChunkMetadataSummary,
    StringChunkMetadataSummary,
)
from .documents import Document
from .types import Chunk, ChunkBatch, ChunkId, InsertedDocMetadata


# Calling this ChunkStore instead of DocumentStore because it stores chunks
# instead of documents.
class ChunkStore(ABC):
    def __init__(self):
        self.summarized_metadata = {}

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

    @abstractmethod
    def documents(self) -> List[dict]:
        raise NotImplementedError

    @abstractmethod
    def context(self, chunk: Chunk, radius: int) -> List[Chunk]:
        raise NotImplementedError

    def _summarize_metadata(
        self,
        key: str,
        metadata_series: pd.Series,
        metadata_type: MetadataType,
        doc_id: int,
        doc_version: int,
    ):
        # summarizing metadata
        if doc_id not in self.summarized_metadata:
            self.summarized_metadata[doc_id] = {}

        if doc_version not in self.summarized_metadata[doc_id]:
            self.summarized_metadata[doc_id][doc_version] = {}

        if metadata_type in [MetadataType.FLOAT, MetadataType.INTEGER]:
            if key not in self.summarized_metadata[doc_id][doc_version]:
                self.summarized_metadata[doc_id][doc_version][key] = (
                    ChunkMetaDataSummary(
                        metadata_type=metadata_type,
                        summary=NumericChunkMetadataSummary(
                            min=metadata_series.min(),
                            max=metadata_series.max(),
                        ),
                    )
                )
            else:
                self.summarized_metadata[doc_id][doc_version][key].summary.min = min(
                    self.summarized_metadata[doc_id][doc_version].summary.min,
                    metadata_series.min(),
                )
                self.summarized_metadata[doc_id][doc_version][key].summary.max = min(
                    self.summarized_metadata[doc_id][doc_version].summary.max,
                    metadata_series.max(),
                )
        else:
            unique_values = set(metadata_series.unique())

            if key in self.summarized_metadata:
                unique_values.add(
                    self.summarized_metadata[doc_id][doc_version][
                        key
                    ].summary.unique_values
                )

            self.summarized_metadata[doc_id][doc_version][key] = ChunkMetaDataSummary(
                metadata_type=metadata_type,
                summary=StringChunkMetadataSummary(
                    unique_values=random.sample(
                        unique_values, k=min(len(unique_values), 100)
                    )  # randomly take 100 unique samples
                ),
            )
