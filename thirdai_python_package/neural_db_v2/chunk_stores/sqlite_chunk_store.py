import operator
import shutil
import uuid
from functools import reduce
from sqlite3 import Connection as SQLite3Connection
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import (
    Column,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    and_,
    create_engine,
    delete,
    event,
    func,
    select,
    text,
    union_all,
)
from sqlalchemy.engine import Engine
from sqlalchemy_utils import StringEncryptedType
from sqlalchemy_utils.types.encrypted.encrypted_type import AesEngine

from ..core.chunk_store import ChunkStore
from ..core.documents import Document
from ..core.types import (
    Chunk,
    ChunkBatch,
    ChunkId,
    InsertedDocMetadata,
    MetadataType,
    pandas_type_mapping,
    sql_type_mapping,
)
from .constraints import Constraint


# In sqlite3, foreign keys are not enabled by default.
# This ensures that sqlite3 connections have foreign keys enabled.
def create_engine_with_fk(database_url, **kwargs):
    engine = create_engine(database_url, **kwargs)

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        if isinstance(dbapi_connection, SQLite3Connection):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON;")
            cursor.close()

    return engine


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
    def __init__(
        self,
        save_path: Optional[str] = None,
        encryption_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        self.db_name = save_path or f"{uuid.uuid4()}.db"
        self.engine = create_engine_with_fk(f"sqlite:///{self.db_name}")

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

        self._create_metadata_tables()

        self.metadata.create_all(self.engine)

        self.next_id = 0

    def _create_metadata_tables(self):
        self.metadata_tables = {}
        for metadata_type, sql_type in sql_type_mapping.items():
            metadata_table = Table(
                f"neural_db_metadata_{metadata_type.value}",
                self.metadata,
                Column(
                    "chunk_id",
                    Integer,
                    ForeignKey("neural_db_chunks.chunk_id", ondelete="CASCADE"),
                    primary_key=True,
                ),
                Column("key", String, primary_key=True),
                Column("value", sql_type, primary_key=True),
                Index(f"ix_metadata_key_value_{metadata_type.value}", "key", "value"),
                extend_existing=True,
            )
            self.metadata_tables[metadata_type] = metadata_table

        self.metadata_type_table = Table(
            "neural_db_metadata_type",
            self.metadata,
            Column("key", String, primary_key=True),
            Column("type", String),
            extend_existing=True,
        )

    def _write_to_table(self, df: pd.DataFrame, table: Table):
        df.to_sql(
            table.name,
            con=self.engine,
            dtype={c.name: c.type for c in table.columns},
            if_exists="append",
            index=False,
        )

    def _store_metadata(self, metadata_col: pd.Series, chunk_ids: pd.Series):
        key = metadata_col.name
        for metadata_type, pd_type in pandas_type_mapping.items():
            if metadata_col.dtype == pd_type:
                metadata_to_insert = []
                for idx, value in metadata_col.items():
                    metadata_to_insert.append(
                        {
                            "chunk_id": int(chunk_ids.iloc[idx]),
                            "key": key,
                            "value": value,
                        }
                    )

                with self.engine.begin() as conn:
                    result = conn.execute(
                        select(self.metadata_type_table.c.type).where(
                            self.metadata_type_table.c.key == key
                        )
                    ).fetchone()

                    if result:
                        existing_type = result.type
                        if existing_type != metadata_type.value:
                            raise ValueError(
                                f"Type mismatch for key '{key}': existing type '{existing_type}', new type '{metadata_type.value}'"
                            )
                    else:
                        conn.execute(
                            self.metadata_type_table.insert().values(
                                key=key, type=metadata_type.value
                            )
                        )

                    conn.execute(
                        self.metadata_tables[metadata_type].insert(), metadata_to_insert
                    )

                continue

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

                self._write_to_table(df=chunk_df, table=self.chunk_table)

                if batch.metadata is not None:
                    singlevalue_metadata, multivalue_metadata = (
                        separate_multivalue_columns(batch.metadata)
                    )
                    for col in singlevalue_metadata:
                        self._store_metadata(singlevalue_metadata[col], chunk_ids)
                    for col in multivalue_metadata:
                        flattened_metadata = flatten_multivalue_column(col, chunk_ids)
                        self._store_metadata(flattened_metadata, chunk_ids)

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

    def get_chunks(self, chunk_ids: List[ChunkId], **kwargs) -> List[Chunk]:
        id_to_chunk = {}

        with self.engine.connect() as conn:
            chunk_stmt = select(self.chunk_table).where(
                self.chunk_table.c.chunk_id.in_(chunk_ids)
            )
            chunk_results = conn.execute(chunk_stmt).fetchall()

            if not chunk_results:
                return []

            for row in chunk_results:
                chunk_id = row.chunk_id
                id_to_chunk[chunk_id] = Chunk(
                    text=row.text,
                    keywords=row.keywords,
                    document=row.document,
                    chunk_id=row.chunk_id,
                    metadata={},
                    doc_id=row.doc_id,
                    doc_version=row.doc_version,
                )

            metadata_subqueries = []
            for _, metadata_table in self.metadata_tables.items():
                subquery = select(
                    metadata_table.c.chunk_id,
                    metadata_table.c.key,
                    metadata_table.c.value,
                ).where(metadata_table.c.chunk_id.in_(chunk_ids))
                metadata_subqueries.append(subquery)

            combined_metadata_query = union_all(*metadata_subqueries).alias(
                "metadata_union"
            )

            metadata_stmt = select(combined_metadata_query)
            metadata_results = conn.execute(metadata_stmt).fetchall()

            for row in metadata_results:
                chunk_id = row.chunk_id
                key = row.key
                value = row.value

                if key in id_to_chunk[chunk_id].metadata:
                    existing_value = id_to_chunk[chunk_id].metadata[key]
                    if isinstance(existing_value, list):
                        existing_value.append(value)
                    else:
                        id_to_chunk[chunk_id].metadata[key] = [existing_value, value]
                else:
                    id_to_chunk[chunk_id].metadata[key] = value

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

        conditions = []
        table_types = set()
        query = None
        with self.engine.begin() as conn:
            for column, constraint in constraints.items():
                result = conn.execute(
                    select(self.metadata_type_table.c.type).where(
                        self.metadata_type_table.c.key == column
                    )
                ).fetchone()

                if result:
                    metadata_type = MetadataType(result.type)
                    table = self.metadata_tables[metadata_type]

                    condition = constraint.sql_condition(
                        column_name=column, table=table
                    )
                    conditions.append(condition)

                    if query is None:
                        query = select(table.c.chunk_id)
                        base_table = table
                    elif metadata_type not in table_types:
                        query = query.select_from(
                            base_table.join(
                                table, base_table.c.chunk_id == table.c.chunk_id
                            )
                        )

                    table_types.add(metadata_type)

        query = query.where(and_(*conditions))
        with self.engine.connect() as conn:
            return set(row.chunk_id for row in conn.execute(query))

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

    def documents(self) -> List[dict]:
        stmt = select(
            self.chunk_table.c.doc_id,
            self.chunk_table.c.doc_version,
            self.chunk_table.c.document,
        ).distinct()

        with self.engine.connect() as conn:
            return [
                {
                    "doc_id": row.doc_id,
                    "doc_version": row.doc_version,
                    "document": row.document,
                }
                for row in conn.execute(stmt)
            ]

    def save(self, path: str):
        shutil.copyfile(self.db_name, path)

    @classmethod
    def load(cls, path: str, encryption_key: Optional[str] = None, **kwargs):
        obj = cls.__new__(cls)

        obj.db_name = path
        obj.engine = create_engine_with_fk(f"sqlite:///{obj.db_name}")

        obj.metadata = MetaData()
        obj.metadata.reflect(bind=obj.engine)

        if "neural_db_chunks" not in obj.metadata.tables:
            raise ValueError("neural_db_chunks table is missing in the database.")

        obj.chunk_table = obj.metadata.tables["neural_db_chunks"]

        if encryption_key:
            obj.chunk_table.columns["text"].type = encrypted_type(encryption_key)
            obj.chunk_table.columns["keywords"].type = encrypted_type(encryption_key)
            obj.chunk_table.columns["document"].type = encrypted_type(encryption_key)

        # Migrate deprecated metadata format
        if "neural_db_metadata" in obj.metadata.tables:
            old_metadata_table = obj.metadata.tables["neural_db_metadata"]
            metadata_columns = [
                col for col in old_metadata_table.columns if col.name != "chunk_id"
            ]

            obj._create_metadata_tables()
            obj.metadata.create_all(obj.engine)

            columns_by_type = {metadata_type: [] for metadata_type in sql_type_mapping}
            key_type_pairs = []
            for col in metadata_columns:
                for metadata_type, sql_type in sql_type_mapping.items():
                    if isinstance(col.type, sql_type):
                        columns_by_type[metadata_type].append(col)
                        key_type_pairs.append(
                            {"key": col.name, "type": metadata_type.value}
                        )

            with obj.engine.connect() as conn:
                for metadata_type, columns in columns_by_type.items():
                    if not columns:
                        continue
                    metadata_table = obj.metadata_tables[metadata_type]
                    select_statements = []
                    for col in columns:
                        stmt = select(
                            [
                                old_metadata_table.c.chunk_id.label("chunk_id"),
                                text(f"'{col.name}'").label("key"),
                                col.label("value"),
                            ]
                        ).where(col != None)
                        select_statements.append(stmt)
                    union_stmt = select_statements[0]
                    for stmt in select_statements[1:]:
                        union_stmt = union_stmt.union_all(stmt)
                    insert_stmt = metadata_table.insert().from_select(
                        ["chunk_id", "key", "value"], union_stmt
                    )
                    conn.execute(insert_stmt)

                insert_stmt = obj.metadata_type_table.insert()
                conn.execute(insert_stmt, key_type_pairs)
        else:
            obj.metadata_tables = {}
            for metadata_type in sql_type_mapping:
                metadata_table_name = f"neural_db_metadata_{metadata_type.value}"
                if metadata_table_name in obj.metadata.tables:
                    obj.metadata_tables[metadata_type] = obj.metadata.tables[
                        metadata_table_name
                    ]
                else:
                    obj._create_metadata_tables()
                    obj.metadata.create_all(obj.engine)

            if "neural_db_metadata_type" in obj.metadata.tables:
                obj.metadata_type_table = obj.metadata.tables["neural_db_metadata_type"]
            else:
                obj._create_metadata_tables()
                obj.metadata.create_all(obj.engine)

        with obj.engine.connect() as conn:
            result = conn.execute(select(func.max(obj.chunk_table.c.chunk_id)))
            max_id = result.scalar()
            obj.next_id = (max_id or 0) + 1

        return obj
