import operator
import os
from functools import reduce
from typing import Dict, Iterable, List, Set, Union

import numpy as np
import pandas as pd
from thirdai.neural_db.utils import pickle_to, unpickle_from

from ..core.chunk_store import ChunkStore
from ..core.types import (
    Chunk,
    ChunkBatch,
    ChunkId,
    CustomIdSupervisedBatch,
    NewChunkBatch,
    SupervisedBatch,
)
from .constraints import Constraint


class PandasChunkStore(ChunkStore):
    def __init__(self, **kwargs):
        super().__init__()

        self.chunk_df = pd.DataFrame()

        self.custom_id_map = {}

        self.metadata_df = pd.DataFrame()

        self.next_id = 0

    def insert(self, chunks: Iterable[NewChunkBatch], **kwargs) -> Iterable[ChunkBatch]:
        all_chunks = [self.chunk_df]
        all_metadata = [self.metadata_df]
        output_batches = []
        for batch in chunks:
            chunk_ids = pd.Series(
                np.arange(self.next_id, self.next_id + len(batch), dtype=np.int64)
            )
            self.next_id += len(batch)

            chunk_df = batch.to_df()
            chunk_df["chunk_id"] = chunk_ids

            all_chunks.append(chunk_df)

            if batch.metadata is not None:
                metadata = batch.metadata.copy(deep=False)
                metadata["chunk_id"] = chunk_ids
                all_metadata.append(metadata)

            if batch.custom_id is not None:
                for custom_id, chunk_id in zip(batch.custom_id, chunk_ids):
                    self.custom_id_map[custom_id] = chunk_id

            output_batches.append(
                ChunkBatch(chunk_id=chunk_ids, text=batch.text, keywords=batch.keywords)
            )

        self.chunk_df = pd.concat(all_chunks)
        self.chunk_df.set_index("chunk_id", inplace=True, drop=False)

        self.metadata_df = pd.concat(all_metadata)

        if not self.metadata_df.empty:
            # Numpy will default missing values to NaN, however we want missing values
            # to be None so that it's consistent with the behavior of sqlalchemy.
            self.metadata_df.replace(to_replace=np.nan, value=None, inplace=True)
            self.metadata_df.set_index("chunk_id", inplace=True, drop=False)

        return output_batches

    def delete(self, chunk_ids: List[ChunkId], **kwargs):
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
                    custom_id=row.custom_id,
                    text=row.text,
                    keywords=row.keywords,
                    metadata=metadata,
                    document=row.document,
                    chunk_id=row.chunk_id,
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

    def _remap_id(self, custom_id: Union[int, str]) -> int:
        if custom_id not in self.custom_id_map:
            raise ValueError(f"Could not find chunk with custom id {custom_id}.")
        return self.custom_id_map[custom_id]

    def remap_custom_ids(
        self, samples: Iterable[CustomIdSupervisedBatch]
    ) -> Iterable[SupervisedBatch]:

        return [
            SupervisedBatch(
                query=batch.query,
                chunk_id=pd.Series(
                    [
                        list(map(self._remap_id, custom_ids))
                        for custom_ids in batch.custom_id
                    ]
                ),
            )
            for batch in samples
        ]

    @staticmethod
    def object_pickle_path(path):
        return os.path.join(path, "object.pkl")

    def save(self, path: str):
        os.makedirs(path)
        pickle_to(self, self.object_pickle_path(path))

    @classmethod
    def load(cls, path: str):
        return unpickle_from(PandasChunkStore.object_pickle_path(path))
