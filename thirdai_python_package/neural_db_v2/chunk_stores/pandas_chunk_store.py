import operator
import os
from functools import reduce
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd
from thirdai.neural_db.utils import pickle_to, unpickle_from

from ..core.chunk_store import ChunkStore
from ..core.documents import Document
from ..core.types import Chunk, ChunkBatch, ChunkId, InsertedDocMetadata
from .constraints import Constraint


class PandasChunkStore(ChunkStore):
    def __init__(self, **kwargs):
        super().__init__()

        self.chunk_df = pd.DataFrame(
            {
                col: []
                for col in [
                    "chunk_id",
                    "custom_id",
                    "text",
                    "keywords",
                    "document",
                    "doc_id",
                    "doc_version",
                ]
            }
        )

        self.doc_versions = {}

        self.metadata_df = pd.DataFrame()

        self.next_id = 0

    def insert(
        self, docs: List[Document], **kwargs
    ) -> Tuple[Iterable[ChunkBatch], List[InsertedDocMetadata]]:
        all_chunks = [self.chunk_df]
        all_metadata = [self.metadata_df]

        output_batches = []
        insert_doc_metadata = []
        for doc in docs:
            doc_id = doc.doc_id()
            doc_version = self.max_version_for_doc(doc_id=doc_id) + 1
            self.doc_versions[doc_id] = doc_version

            doc_chunk_ids = []
            for batch in doc.chunks():
                chunk_ids = pd.Series(
                    np.arange(self.next_id, self.next_id + len(batch), dtype=np.int64)
                )
                self.next_id += len(batch)
                doc_chunk_ids.extend(chunk_ids)

                chunk_df = batch.to_df()
                chunk_df["chunk_id"] = chunk_ids
                chunk_df["doc_id"] = doc_id
                chunk_df["doc_version"] = doc_version

                all_chunks.append(chunk_df)

                if batch.metadata is not None:
                    metadata = batch.metadata.copy(deep=False)
                    metadata["chunk_id"] = chunk_ids
                    all_metadata.append(metadata)

                output_batches.append(
                    ChunkBatch(
                        chunk_id=chunk_ids, text=batch.text, keywords=batch.keywords
                    )
                )

            insert_doc_metadata.append(
                InsertedDocMetadata(
                    doc_id=doc_id, doc_version=doc_version, chunk_ids=doc_chunk_ids
                )
            )

        self.chunk_df = pd.concat(all_chunks)
        self.chunk_df.set_index("chunk_id", inplace=True, drop=False)

        self.metadata_df = pd.concat(all_metadata)

        if not self.metadata_df.empty:
            # Numpy will default missing values to NaN, however we want missing values
            # to be None so that it's consistent with the behavior of sqlalchemy.
            self.metadata_df.replace(to_replace=np.nan, value=None, inplace=True)
            self.metadata_df.set_index("chunk_id", inplace=True, drop=False)

        return output_batches, insert_doc_metadata

    def delete(self, chunk_ids: List[ChunkId]):
        self.chunk_df.drop(chunk_ids, inplace=True)
        if not self.metadata_df.empty:
            self.metadata_df.drop(chunk_ids, inplace=True)

    def get_chunks(self, chunk_ids: List[ChunkId], **kwargs) -> List[Chunk]:
        try:
            chunks = self.chunk_df.loc[chunk_ids]
            metadatas = (
                self.metadata_df.loc[chunk_ids] if not self.metadata_df.empty else None
            )
        except KeyError:
            raise ValueError(
                f"Could not find chunk with one or more ids in {chunk_ids}."
            )
        output_chunks = []
        for i, row in enumerate(chunks.itertuples()):
            if metadatas is not None:
                metadata = metadatas.iloc[i].to_dict()
                del metadata["chunk_id"]
            else:
                metadata = None
            output_chunks.append(
                Chunk(
                    text=row.text,
                    keywords=row.keywords,
                    metadata=metadata,
                    document=row.document,
                    chunk_id=row.chunk_id,
                    doc_id=row.doc_id,
                    doc_version=row.doc_version,
                )
            )
        return output_chunks

    def filter_chunk_ids(
        self, constraints: Dict[str, Constraint], **kwargs
    ) -> Set[ChunkId]:
        if not len(constraints):
            raise ValueError("Cannot call filter_chunk_ids with empty constraints.")

        if self.metadata_df.empty:
            raise ValueError("Cannot filter constraints with no metadata.")

        condition = reduce(
            operator.and_,
            [
                constraint.pd_filter(column_name=column, df=self.metadata_df)
                for column, constraint in constraints.items()
            ],
        )

        return set(self.chunk_df[condition]["chunk_id"])

    def get_doc_chunks(self, doc_id: str, before_version: int) -> Set[ChunkId]:
        return self.chunk_df["chunk_id"][
            (self.chunk_df["doc_id"] == doc_id)
            & (self.chunk_df["doc_version"] < before_version)
        ].to_list()

    def max_version_for_doc(self, doc_id: str) -> int:
        return self.doc_versions.get(doc_id, 0)

    def save(self, path: str):
        pickle_to(self, path)

    @classmethod
    def load(cls, path: str):
        return unpickle_from(path)
