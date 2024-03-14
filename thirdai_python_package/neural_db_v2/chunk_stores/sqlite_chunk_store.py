from ..core.chunk_store import ChunkStore
from typing import Iterable, List, Set
from ..core.types import (
    ChunkBatch,
    NewChunkBatch,
    ChunkId,
    Chunk,
    CustomIdSupervisedBatch,
    SupervisedBatch,
)
from sqlalchemy import create_engine
import uuid
from sqlalchemy import Table, Column, Integer, String, MetaData, select, delete
import pandas as pd
import numpy as np


class SQLiteChunkStore(ChunkStore):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.db_name = f"{uuid.uuid4()}.db"

        self.engine = create_engine(f"sqlite:///:memory:{self.db_name}")
        self.metadata = MetaData()

        self.chunk_table = Table(
            "neural_db_chunks",
            self.metadata,
            Column("chunk_id", Integer, primary_key=True),
            Column("text", String),
            Column("keywords", String),
            Column("document", String),
        )
        self.metadata.create_all(self.engine)

        self.custom_id_table = None

        self.next_id = 0

    def create_custom_id_table(self, integer_custom_ids):
        custom_id_dtype = Integer if integer_custom_ids else String
        self.custom_id_table = Table(
            "neural_db_custom_ids",
            self.metadata,
            Column("custom_id", custom_id_dtype, primary_key=True),
            Column("chunk_id", Integer),
        )
        self.metadata.create_all(self.engine)

    def _write_to_table(self, df: pd.DataFrame, table: Table):
        df.to_sql(
            table.name,
            con=self.engine,
            dtype={c.name: c.type for c in table.columns},
            if_exists="append",
            index=False,
        )

    def insert(self, chunks: Iterable[NewChunkBatch], **kwargs) -> Iterable[ChunkBatch]:
        inserted_batches = []
        for batch in chunks:
            chunk_ids = pd.Series(
                np.arange(self.next_id, self.next_id + len(batch.text), dtype=np.int64)
            )

            chunk_df = batch.to_df(with_custom_id=False, with_metadata=False)
            chunk_df["chunk_id"] = chunk_ids

            if batch.custom_id is not None:
                batch_integer_custom_ids = batch.custom_id.dtype == int
                if self.custom_id_table is None:
                    self.create_custom_id_table(
                        integer_custom_ids=batch_integer_custom_ids
                    )

                table_integer_custom_ids = isinstance(
                    self.custom_id_table.columns.custom_id.type, Integer
                )

                if table_integer_custom_ids != batch_integer_custom_ids:
                    raise ValueError(
                        "Custom ids must all have the same type. Found some custom ids with type int, and some with type str."
                    )

                custom_id_df = pd.DataFrame(
                    {"custom_id": batch.custom_id, "chunk_id": chunk_ids}
                )
                self._write_to_table(df=custom_id_df, table=self.custom_id_table)

            self._write_to_table(df=chunk_df, table=self.chunk_table)

            self.next_id += len(batch.text)

            inserted_batches.append(
                ChunkBatch(
                    chunk_id=chunk_ids,
                    text=batch.text,
                    keywords=batch.keywords,
                )
            )

        return inserted_batches

    def delete(self, chunk_ids: List[ChunkId], **kwargs):
        stmt = delete(self.chunk_table).where(
            self.chunk_table.c.chunk_id.in_(chunk_ids)
        )
        with self.engine.begin() as conn:
            conn.execute(stmt)

    def get_chunks(self, chunk_ids: List[ChunkId], **kwargs) -> List[Chunk]:
        stmt = select(self.chunk_table).where(
            self.chunk_table.c.chunk_id.in_(chunk_ids)
        )

        id_to_chunk = {}
        with self.engine.connect() as conn:
            for row in conn.execute(stmt):
                id_to_chunk[row.chunk_id] = Chunk(
                    custom_id=None,
                    text=row.text,
                    keywords=row.keywords,
                    document=row.document,
                    chunk_id=row.chunk_id,
                    metadata=None,
                )

        chunks = []
        for chunk_id in chunk_ids:
            if chunk_id not in id_to_chunk:
                raise ValueError(f"Could not find chunk with id {chunk_id}.")
            chunks.append(id_to_chunk[chunk_id])

        return chunks

    def filter_chunk_ids(self, constraints: dict, **kwargs) -> Set[ChunkId]:
        pass

    def remap_custom_ids(
        self, samples: Iterable[CustomIdSupervisedBatch]
    ) -> Iterable[SupervisedBatch]:
        remapped_batches = []

        for batch in samples:
            chunk_ids = []
            with self.engine.connect() as conn:
                for custom_ids in batch.custom_id:
                    sample_ids = []
                    for custom_id in custom_ids:
                        stmt = select(self.custom_id_table.c.chunk_id).where(
                            self.custom_id_table.c.custom_id == custom_id
                        )
                        if result := conn.execute(stmt).first():
                            sample_ids.append(result.chunk_id)
                        else:
                            raise ValueError(f"Could not find custom id {custom_id}.")
                    chunk_ids.append(sample_ids)

            remapped_batches.append(
                SupervisedBatch(query=batch.query, chunk_id=pd.Series(chunk_ids))
            )

        return remapped_batches
