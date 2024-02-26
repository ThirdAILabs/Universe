from typing import List, Set, Iterable

import pandas as pd
from core.types import ChunkId, Chunk, NewChunkBatch, ChunkBatch
from core.chunk_store import ChunkStore


class DataFrameChunkStore(ChunkStore):
    def __init__(self):
        self.text_df = pd.DataFrame({"chunk_id": [], "text": [], "keywords": []})
        self.text_df = self.text_df.set_index("chunk_id")
        self.metadata_df = pd.DataFrame({"chunk_id": [], "key": [], "value": []})
        self.metadata_df = self.metadata_df.set_index("chunk_id")
        self.custom_id_to_chunk_id = {}
        self.chunk_id_to_custom_id = {}
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
                    "text": batch.text,
                    "keywords": batch.keywords,
                }
            ).set_index("chunk_id")

            metadata_df_delta = pd.DataFrame.from_records(
                [
                    {"chunk_id": chunk_id, "key": key, "value": value}
                    for chunk_id, metadata in zip(chunk_ids, batch.metadata)
                    for key, value in metadata.items()
                ]
            ).set_index("chunk_id")

            if batch.custom_id:
                for chunk_id, custom_id in zip(chunk_ids, batch.custom_id):
                    self.custom_id_to_chunk_id[custom_id] = chunk_id
                    self.chunk_id_to_custom_id[chunk_id] = custom_id

            yield ChunkBatch(
                custom_id=batch.custom_id,
                text=batch.text,
                keywords=batch.keywords,
                metadata=batch.metadata,
                document=batch.document,
                chunk_id=chunk_ids,
            )

            self.text_df = pd.concat([self.text_df, text_df_delta])
            self.metadata_df = pd.concat([self.metadata_df, metadata_df_delta])

    def delete(self, chunk_id: ChunkId, **kwargs):
        self.text_df.drop([chunk_id])
        self.metadata_df.drop([chunk_id])

    def delete_batch(self, chunk_ids: List[ChunkId], **kwargs):
        self.text_df.drop(chunk_ids)
        self.metadata_df.drop(chunk_ids)

    def _get_chunk(self, chunk_id: ChunkId, **kwargs) -> Chunk:
        text = self.text_df.loc[chunk_id]
        metadata = self.metadata_df.loc[chunk_id : chunk_id + 1]
        return Chunk(
            chunk_id=chunk_id,
            custom_id=self.chunk_id_to_custom_id[chunk_id],
            text=text["text"],
            keywords=text["keywords"],
            metadata={
                key: value for key, value in zip(metadata["key"], metadata["value"])
            },
        )

    def get_chunks(self, chunk_ids: List[ChunkId], **kwargs) -> List[Chunk]:
        # This is a very inefficient implementation. This is POC code.
        return [self._get_chunk(chunk_id) for chunk_id in chunk_ids]

    def filter_chunk_ids(self, constraints: dict, **kwargs) -> Set[ChunkId]:
        pass
