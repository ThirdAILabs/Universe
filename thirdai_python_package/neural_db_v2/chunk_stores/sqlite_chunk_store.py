import operator
import shutil
import uuid
from functools import reduce
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import (
    Column,
    Engine,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    delete,
    func,
    join,
    select,
    text,
)
from sqlalchemy_utils import StringEncryptedType
from sqlalchemy_utils.types.encrypted.encrypted_type import AesEngine

from ..core.chunk_store import ChunkStore
from ..core.documents import Document
from ..core.types import Chunk, ChunkBatch, ChunkId, InsertedDocMetadata
from .constraints import Constraint


def separate_multivalue_columns(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[pd.Series]]:
    multivalue_columns = []
    for col in df.columns:
        if df[col].dtype == object and df[col].map(lambda x: isinstance(x, List)).any():
            multivalue_columns.append(df[col])

    return df.drop([c.name for c in multivalue_columns], axis=1), multivalue_columns


def flatten_multivalue_column(column: pd.Series, chunk_ids: pd.Series) -> pd.DataFrame:
    return (
        pd.DataFrame({"chunk_id": chunk_ids, column.name: column})
        .explode(column.name)  # flattens column and repeats values in other column
        .dropna()  # explode converts [] to a row with a NaN in the exploded column
        .reset_index(drop=True)  # explode repeats index values, this resets that
        .infer_objects(copy=False)  # explode doesn't adjust dtype of exploded column
    )


MULTIVALUE_METADATA_PREFIX = "multivalue_metadata_"


def get_sql_type(name, dtype):
    if dtype == int:
        return Integer
    elif dtype == float:
        return Float
    elif dtype == object:
        return String
    else:
        raise ValueError(
            f"Column {name} has dtype {str(dtype)} which is not a supported type for metadata columns."
        )


def get_sql_columns(df: pd.DataFrame):
    return [Column(col, get_sql_type(col, df[col].dtype)) for col in df.columns]


class SqlLiteIterator:
    def __init__(
        self,
        table: Table,
        engine: Engine,
        min_insertion_chunk_id: int,
        max_insertion_chunk_id: int,
        max_in_memory_batches: int = 100,
    ):
        self.chunk_table = table
        self.engine = engine

        # Since assigned chunk_ids are contiguous, each SqlLiteIterator can search
        # through a range of chunk_ids. We need a min and a max in the case
        # we do an insertion while another iterator instance still exists
        self.min_insertion_chunk_id = min_insertion_chunk_id
        self.max_insertion_chunk_id = max_insertion_chunk_id

        self.max_in_memory_batches = max_in_memory_batches

    def __next__(self) -> Optional[ChunkBatch]:
        # The "next" call on the sql_row_iterator returns one row at a time
        # despite fetching them in "max_in_memory_batches" quantities from the database.
        # Thus we call "next" "max_in_memory_batches" times to pull out all the rows we want
        sql_lite_batch = []
        try:
            for _ in range(self.max_in_memory_batches):
                sql_lite_batch.append(next(self.sql_row_iterator))
        except StopIteration:
            if not sql_lite_batch:
                raise StopIteration

        df = pd.DataFrame(sql_lite_batch, columns=self.sql_row_iterator.keys())

        return ChunkBatch(
            chunk_id=df["chunk_id"],
            text=df["text"],
            keywords=df["keywords"],
        )

    def __iter__(self):
        stmt = select(self.chunk_table).where(
            (self.chunk_table.c.chunk_id >= self.min_insertion_chunk_id)
            & (self.chunk_table.c.chunk_id < self.max_insertion_chunk_id)
        )
        with self.engine.connect() as conn:
            result = conn.execute(stmt)
            self.sql_row_iterator = result.yield_per(self.max_in_memory_batches)
        return self


def encrypted_type(key: str):
    return StringEncryptedType(String, key=key, engine=AesEngine)


class SQLiteChunkStore(ChunkStore):
    def __init__(self, encryption_key: Optional[str] = None, **kwargs):
        super().__init__()

        self.db_name = f"{uuid.uuid4()}.db"
        self.engine = create_engine(f"sqlite:///{self.db_name}")

        self.metadata = MetaData()

        text_type = encrypted_type(encryption_key) if encryption_key else String

        self.chunk_table = Table(
            "neural_db_chunks",
            self.metadata,
            Column("chunk_id", Integer, primary_key=True),
            Column("text", text_type),
            Column("keywords", text_type),
            Column("document", text_type),
            Column("doc_id", String, index=True),
            Column("doc_version", Integer),
        )

        self.metadata.create_all(self.engine)

        self.metadata_table = None

        self.multivalue_metadata_tables = {}

        self.next_id = 0

    def _write_to_table(self, df: pd.DataFrame, table: Table):
        df.to_sql(
            table.name,
            con=self.engine,
            dtype={c.name: c.type for c in table.columns},
            if_exists="append",
            index=False,
        )

    def _add_metadata_column(self, column: Column):
        column_name = column.compile(dialect=self.engine.dialect)
        column_type = column.type.compile(self.engine.dialect)
        stmt = text(
            f"ALTER TABLE {self.metadata_table.name} ADD COLUMN {column_name} {column_type}"
        )

        with self.engine.begin() as conn:
            conn.execute(stmt)

        # This is so that sqlalchemy recognizes the new column.
        self.metadata = MetaData()
        self.metadata.reflect(bind=self.engine)
        self.metadata_table = Table(
            self.metadata_table.name, self.metadata, autoload_with=self.engine
        )

    def _store_singlevalue_metadata(self, metadata: pd.DataFrame, chunk_ids: pd.Series):
        metadata_columns = get_sql_columns(metadata)
        if self.metadata_table is None:
            self.metadata_table = Table(
                "neural_db_metadata",
                self.metadata,
                Column("chunk_id", Integer, primary_key=True),
                *metadata_columns,
            )
            self.metadata.create_all(self.engine)
        else:
            for column in metadata_columns:
                if column.name not in self.metadata_table.columns:
                    self._add_metadata_column(column=column)
                else:
                    if str(column.type) != str(
                        self.metadata_table.columns[column.name].type
                    ):
                        raise ValueError(
                            f"Existing metadata for column {column.name} has type {str(self.metadata_table.columns[column.name].type)} but new metadata has type {str(column.type)}."
                        )
        metadata["chunk_id"] = chunk_ids
        self._write_to_table(df=metadata, table=self.metadata_table)

    def _store_multivalue_metadata(self, metadata_col: pd.Series, chunk_ids: pd.Series):
        flattened_metadata = flatten_multivalue_column(metadata_col, chunk_ids)

        if metadata_col.name not in self.multivalue_metadata_tables:
            table = Table(
                MULTIVALUE_METADATA_PREFIX + metadata_col.name,
                self.metadata,
                Column("chunk_id", Integer, index=True, primary_key=True),
                Column(
                    metadata_col.name,
                    get_sql_type(
                        metadata_col.name, flattened_metadata[metadata_col.name].dtype
                    ),
                    primary_key=True,
                ),
            )
            self.metadata.create_all(self.engine)
            self.multivalue_metadata_tables[metadata_col.name] = table

        self._write_to_table(
            flattened_metadata, self.multivalue_metadata_tables[metadata_col.name]
        )

    def insert(
        self, docs: List[Document], max_in_memory_batches=10000, **kwargs
    ) -> Tuple[Iterable[ChunkBatch], List[InsertedDocMetadata]]:
        min_insertion_chunk_id = self.next_id

        inserted_doc_metadata = []
        for doc in docs:
            doc_id = doc.doc_id()
            doc_version = self.max_version_for_doc(doc_id) + 1

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

                if batch.metadata is not None:
                    singlevalue_metadata, multivalue_metadata = (
                        separate_multivalue_columns(batch.metadata)
                    )
                    self._store_singlevalue_metadata(singlevalue_metadata, chunk_ids)
                    for col in multivalue_metadata:
                        self._store_multivalue_metadata(col, chunk_ids)

                self._write_to_table(df=chunk_df, table=self.chunk_table)

            inserted_doc_metadata.append(
                InsertedDocMetadata(
                    doc_id=doc_id, doc_version=doc_version, chunk_ids=doc_chunk_ids
                )
            )

        max_insertion_chunk_id = self.next_id

        inserted_chunks_iterator = SqlLiteIterator(
            table=self.chunk_table,
            engine=self.engine,
            min_insertion_chunk_id=min_insertion_chunk_id,
            max_insertion_chunk_id=max_insertion_chunk_id,
            max_in_memory_batches=max_in_memory_batches,
        )

        return inserted_chunks_iterator, inserted_doc_metadata

    def delete(self, chunk_ids: List[ChunkId]):
        with self.engine.begin() as conn:
            delete_chunks = delete(self.chunk_table).where(
                self.chunk_table.c.chunk_id.in_(chunk_ids)
            )
            conn.execute(delete_chunks)

            if self.metadata_table is not None:
                delete_metadata = delete(self.metadata_table).where(
                    self.metadata_table.c.chunk_id.in_(chunk_ids)
                )
                conn.execute(delete_metadata)

    def get_chunks(self, chunk_ids: List[ChunkId], **kwargs) -> List[Chunk]:
        id_to_chunk = {}

        with self.engine.connect() as conn:
            chunk_stmt = select(self.chunk_table).where(
                self.chunk_table.c.chunk_id.in_(chunk_ids)
            )
            for row in conn.execute(chunk_stmt).all():
                id_to_chunk[row.chunk_id] = Chunk(
                    text=row.text,
                    keywords=row.keywords,
                    document=row.document,
                    chunk_id=row.chunk_id,
                    metadata=None,
                    doc_id=row.doc_id,
                    doc_version=row.doc_version,
                )

            if self.metadata_table is not None:
                metadata_stmt = select(self.metadata_table).where(
                    self.metadata_table.c.chunk_id.in_(chunk_ids)
                )
                for row in conn.execute(metadata_stmt).all():
                    metadata = row._asdict()
                    del metadata["chunk_id"]
                    id_to_chunk[row.chunk_id].metadata = metadata

            for key, table in self.multivalue_metadata_tables.items():
                multivalue_stmt = select(table).where(table.c.chunk_id.in_(chunk_ids))
                for row in conn.execute(multivalue_stmt).all():
                    if id_to_chunk[row.chunk_id].metadata is None:
                        id_to_chunk[row.chunk_id].metadata = {}

                    if key not in id_to_chunk[row.chunk_id].metadata:
                        id_to_chunk[row.chunk_id].metadata[key] = []

                    id_to_chunk[row.chunk_id].metadata[key].append(getattr(row, key))

        chunks = []
        for chunk_id in chunk_ids:
            if chunk_id not in id_to_chunk:
                raise ValueError(f"Could not find chunk with id {chunk_id}.")
            chunks.append(id_to_chunk[chunk_id])

        return chunks

    def filter_chunk_ids(
        self, constraints: Dict[str, Constraint], **kwargs
    ) -> Set[ChunkId]:
        if not len(constraints):
            raise ValueError("Cannot call filter_chunk_ids with empty constraints.")

        if self.metadata_table is None:
            raise ValueError("Cannot filter constraints with no metadata.")

        select_from = self.metadata_table
        conditions = []
        for column, constraint in constraints.items():
            if column in self.multivalue_metadata_tables:
                table = self.multivalue_metadata_tables[column]
                conditions.append(
                    constraint.sql_condition(column_name=column, table=table)
                )
                select_from = join(
                    select_from,
                    table,
                    self.metadata_table.c.chunk_id == table.c.chunk_id,
                )
            else:
                conditions.append(
                    constraint.sql_condition(
                        column_name=column, table=self.metadata_table
                    )
                )

        stmt = (
            select(self.metadata_table.c.chunk_id)
            .select_from(select_from)
            .where(reduce(operator.and_, conditions))
        )

        with self.engine.connect() as conn:
            return set(row.chunk_id for row in conn.execute(stmt))

    def get_doc_chunks(self, doc_id: str, before_version: int) -> List[ChunkId]:
        stmt = select(self.chunk_table.c.chunk_id).where(
            (self.chunk_table.c.doc_id == doc_id)
            & (self.chunk_table.c.doc_version < before_version)
        )

        with self.engine.connect() as conn:
            return [row.chunk_id for row in conn.execute(stmt)]

    def max_version_for_doc(self, doc_id: str) -> int:
        stmt = select(func.max(self.chunk_table.c.doc_version)).where(
            self.chunk_table.c.doc_id == doc_id
        )

        with self.engine.connect() as conn:
            result = conn.execute(stmt)
            return result.scalar() or 0

    def save(self, path: str):
        shutil.copyfile(self.db_name, path)

    @classmethod
    def load(cls, path: str, encryption_key: Optional[str] = None):
        obj = cls.__new__(cls)

        obj.db_name = path
        obj.engine = create_engine(f"sqlite:///{obj.db_name}")

        obj.metadata = MetaData()
        obj.metadata.reflect(bind=obj.engine)

        if "neural_db_chunks" not in obj.metadata.tables:
            raise ValueError("neural_db_chunks table is missing in the database.")

        obj.chunk_table = obj.metadata.tables["neural_db_chunks"]

        if encryption_key:
            obj.chunk_table.columns["text"].type = encrypted_type(encryption_key)
            obj.chunk_table.columns["keywords"].type = encrypted_type(encryption_key)
            obj.chunk_table.columns["document"].type = encrypted_type(encryption_key)

        if "neural_db_metadata" in obj.metadata.tables:
            obj.metadata_table = obj.metadata.tables["neural_db_metadata"]
        else:
            obj.metadata_table = None

        obj.multivalue_metadata_tables = {}
        for name, table in obj.metadata.tables.items():
            if name.startswith(MULTIVALUE_METADATA_PREFIX):
                obj.multivalue_metadata_tables[
                    name[len(MULTIVALUE_METADATA_PREFIX) :]
                ] = table

        with obj.engine.connect() as conn:
            result = conn.execute(select(func.max(obj.chunk_table.c.chunk_id)))
            max_id = result.scalar()
            obj.next_id = (max_id or 0) + 1

            chunk_count = conn.execute(
                select(func.count()).select_from(obj.chunk_table)
            ).scalar()

        return obj
