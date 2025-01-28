import operator
from collections import defaultdict
from functools import reduce
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd
from thirdai.neural_db.utils import pickle_to, unpickle_from

from ..core.chunk_store import ChunkStore
from ..core.documents import Document
from ..core.types import (
    Chunk,
    ChunkBatch,
    ChunkId,
    InsertedDocMetadata,
    pandas_type_to_metadata_type,
)
from .constraints import Constraint
from .document_metadata_summary import DocumentMetadataSummary


class PandasChunkStore(ChunkStore):
    def __init__(self, **kwargs):
        super().__init__()

        self.chunk_df = pd.DataFrame(
            {
                "chunk_id": pd.Series(dtype="int64"),
                "custom_id": pd.Series(dtype="object"),
                "text": pd.Series(dtype="object"),
                "keywords": pd.Series(dtype="object"),
                "document": pd.Series(dtype="object"),
                "doc_id": pd.Series(dtype="object"),
                "doc_version": pd.Series(dtype="int64"),
            }
        )

        self.doc_versions = {}

        self.metadata_df = pd.DataFrame()

        self.next_id = 0

        self.document_metadata_summary = DocumentMetadataSummary()

    def insert(
        self, docs: List[Document], **kwargs
    ) -> Tuple[Iterable[ChunkBatch], List[InsertedDocMetadata]]:
        all_chunks = [self.chunk_df]
        all_metadata = [self.metadata_df]
        new_metadata_keys = defaultdict(set)

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

                    # Collect the metadata
                    new_metadata_keys[(doc_id, doc_version)].update(
                        metadata.columns.tolist()
                    )

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

            # Unlike SqliteChunkStore, PandasChunkStore does not raise an error when two documents have the same metadata key
            # with different data types. As a result, the summarized metadata for such keys in PandasChunkStore might be incorrect.
            # TODO(anyone): Throw the error in such case, and bind metadata-key with it's (doc_id, doc_version).

            # summarize the metadata
            for (doc_id, doc_version), metadata_keys in new_metadata_keys.items():
                for key in metadata_keys:
                    metadata_type = pandas_type_to_metadata_type.get(
                        self.metadata_df[key].dropna().infer_objects().dtype,
                        None,  # first DropNaN because we would have NaN in rows for the other document's metadata key, then infer the type again
                    )
                    if metadata_type:
                        self.document_metadata_summary.summarize_metadata(
                            key,
                            self.metadata_df[key],
                            metadata_type,
                            doc_id,
                            doc_version,
                            overwrite_type=True,
                        )

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
                metadata = metadatas.iloc[i].dropna().to_dict()
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

        missing_columns = [
            column for column in constraints if column not in self.metadata_df.columns
        ]
        if missing_columns:
            raise KeyError(f"Missing columns in metadata: {', '.join(missing_columns)}")

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

    def documents(self) -> List[dict]:
        return (
            self.chunk_df[["doc_id", "doc_version", "document"]]
            .drop_duplicates()
            .to_dict("records")
        )

    def context(self, chunk: Chunk, radius: int) -> List[Chunk]:
        rows = self.chunk_df[
            (self.chunk_df["chunk_id"] >= (chunk.chunk_id - radius))
            & (self.chunk_df["chunk_id"] <= (chunk.chunk_id + radius))
            & (self.chunk_df["doc_id"] == chunk.doc_id)
            & (self.chunk_df["doc_version"] == chunk.doc_version)
        ]

        return [
            Chunk(
                text=row.text,
                keywords=row.keywords,
                metadata=None,
                document=row.document,
                chunk_id=row.chunk_id,
                doc_id=row.doc_id,
                doc_version=row.doc_version,
            )
            for row in rows.itertuples()
        ]

    def save(self, path: str):
        pickle_to(self, path)

    @classmethod
    def load(cls, path: str, **kwargs):
        return unpickle_from(path)
