from typing import List, Set, Iterable

import pandas as pd
from core.types import ChunkId, Chunk, NewChunkBatch, ChunkBatch
from core.chunk_store import ChunkStore


class DataFrameChunkStore(ChunkStore):
    def __init__(self):
        self.text_df = pd.DataFrame(
            {"chunk_id": [], "text": [], "keywords": [], "metadata": []}
        )
        self.text_df = self.text_df.set_index("chunk_id")
        self.custom_id_to_chunk_id = {}
        self.last_id = 0

    def insert_batch(
        self,
        chunks: Iterable[NewChunkBatch],
        **kwargs,
    ) -> Iterable[ChunkBatch]:
        for batch in chunks:
            new_last_id = self.last_id + len(batch)
            chunk_ids = pd.Series(range(self.last_id, new_last_id))
            self.last_id = new_last_id

            text_df_delta = pd.DataFrame(
                {
                    "chunk_id": chunk_ids,
                    "custom_id": batch.custom_id or [None for _ in chunk_ids],
                    "text": batch.text,
                    "keywords": batch.keywords,
                    "metadata": batch.metadata,
                }
            ).set_index("chunk_id")

            if batch.custom_id:
                for chunk_id, custom_id in zip(chunk_ids, batch.custom_id):
                    self.custom_id_to_chunk_id[custom_id] = chunk_id

            yield ChunkBatch(
                custom_id=batch.custom_id,
                text=batch.text,
                keywords=batch.keywords,
                metadata=batch.metadata,
                document=batch.document,
                chunk_id=chunk_ids,
            )

            self.text_df = pd.concat([self.text_df, text_df_delta])

    def delete(self, chunk_ids: List[ChunkId], **kwargs):
        self.text_df.drop(chunk_ids)
        self.metadata_df.drop(chunk_ids)

    def get_chunks(self, chunk_ids: List[ChunkId], **kwargs) -> List[Chunk]:
        return [Chunk(**args) for args in self.text_df.loc[chunk_ids].to_dict()]

    def filter_chunk_ids(self, constraints: dict, **kwargs) -> Set[ChunkId]:
        pass
