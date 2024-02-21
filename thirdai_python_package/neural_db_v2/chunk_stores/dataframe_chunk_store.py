from typing import Iterable, List, Set

import pandas as pd
from core.types import ChunkId, NewChunk

from thirdai_python_package.neural_db_v2.core.chunk_store import ChunkStore


class DataFrameChunkStore(ChunkStore):
    def __init__(self):
        self.text_df = pd.DataFrame({"chunk_id": [], "text": [], "keywords": []})
        self.text_df = self.text_df.set_index("chunk_id")
        self.metadata_df = pd.DataFrame({"chunk_id": [], "key": [], "value": []})
        self.metadata_df = self.metadata_df.set_index("chunk_id")

    def insert_batch(
        self,
        chunks: Iterable[NewChunk],
        **kwargs,
    ):
        prev_max_id = self.text_df.index.max()
        chunk_ids = [prev_max_id + i for i in range(len(chunks))]
        text_df_delta = pd.DataFrame.from_records(
            [
                {"chunk_id": chunk_id, "text": doc.text, "keywords": doc.keywords}
                for chunk_id, doc in zip(chunk_ids, chunks)
            ]
        ).set_index("chunk_id")

        metadata_df_delta = pd.DataFrame.from_records(
            [
                {"chunk_id": chunk_id, "key": key, "value": value}
                for chunk_id, doc in zip(chunk_ids, chunks)
                for key, value in doc.metadata.items()
            ]
        ).set_index("chunk_id")

        self.text_df = pd.concat([self.text_df, text_df_delta])
        self.metadata_df = pd.concat([self.metadata_df, metadata_df_delta])

    def delete(self, chunk_id: ChunkId, **kwargs):
        self.text_df.drop([chunk_id])
        self.metadata_df.drop([chunk_id])

    def delete_batch(self, chunk_ids: List[ChunkId], **kwargs):
        self.text_df.drop(chunk_ids)
        self.metadata_df.drop(chunk_ids)

    def get_chunk(self, chunk_id: ChunkId, **kwargs):
        text = self.text_df.loc[chunk_id]
        metadata = self.metadata_df.loc[chunk_id : chunk_id + 1]
        return NewChunk(
            chunk_id=chunk_id,
            text=text["text"],
            keywords=text["keywords"],
            metadata={
                key: value for key, value in zip(metadata["key"], metadata["value"])
            },
        )

    def get_chunk_batch(self, chunk_ids: List[ChunkId], **kwargs):
        # This is a very inefficient implementation. This is POC code.
        return [self.get_chunk(chunk_id) for chunk_id in chunk_ids]

    def matching_chunk_ids(self, constraints: dict, **kwargs) -> Set[ChunkId]:
        pass
