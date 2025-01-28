import itertools
import random
import shutil
import uuid
from collections import defaultdict
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
    alias,
    and_,
    create_engine,
    delete,
    event,
    func,
    select,
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
    ChunkMetaDataSummary,
    InsertedDocMetadata,
    MetadataType,
    NumericChunkMetadataSummary,
    StringChunkMetadataSummary,
    pandas_type_to_metadata_type,
    sql_type_mapping,
)
from .constraints import Constraint
from .document_metadata_summary import DocumentMetadataSummary


# In sqlite3, foreign keys are not enabled by default.
# This ensures that sqlite3 connections have foreign keys enabled.
def create_engine_with_fk(
    database_url,
    pool_size=20,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1200,
    **kwargs,
):
    engine = create_engine(
        database_url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_timeout=pool_timeout,
        pool_recycle=pool_recycle,
        **kwargs,
    )

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
    df = (
        pd.DataFrame({"chunk_id": chunk_ids, column.name: column})
        .explode(column.name)  # flattens column and repeats values in other column
        .dropna()  # explode converts [] to a row with a NaN in the exploded column
        .reset_index(drop=True)  # explode repeats index values, this resets that
        .infer_objects(copy=False)  # explode doesn't adjust dtype of exploded column
    )
    return df


def sqlite_insert_bulk(table, conn, keys, data_iter):
    columns = ", ".join([f'"{k}"' for k in keys])
    placeholders = ", ".join(["?"] * len(keys))
    insert_stmt = f"INSERT INTO {table.name} ({columns}) VALUES ({placeholders})"

    dbapi_conn = conn.connection
    cursor = dbapi_conn.cursor()

    try:
        while True:
            chunk = list(itertools.islice(data_iter, 10000))
            if not chunk:
                break
            cursor.executemany(insert_stmt, chunk)

    except Exception as e:
        dbapi_conn.rollback()
        raise e
    finally:
        cursor.close()


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
        use_metadata_index: bool = False,
        **kwargs,
    ):
        """
        Params:
            save_path: Optional[str] - Path to save db to, otherwise is random
            encryption_key: Optional[str] - Must be passed to encrypt data
            use_metadata_index: bool - If true, insertion time doubles but query time with constraints roughly halves
        """
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

        self._create_metadata_tables(use_metadata_index)

        self.metadata.create_all(self.engine)

        self.next_id = 0

        self.document_metadata_summary = DocumentMetadataSummary()

    def _create_metadata_tables(self, use_metadata_index: bool = False):
        self.metadata_tables = {}
        for metadata_type, sql_type in sql_type_mapping.items():
            if use_metadata_index:
                metadata_index = Index(
                    f"ix_metadata_key_value_{metadata_type.value}", "key", "value"
                )
            else:
                metadata_index = Index(f"ix_metadata_key_{metadata_type.value}", "key")

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
                metadata_index,
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

    def _write_to_table(
        self, df: pd.DataFrame, table: Table, con=None, bulk_insert=False
    ):
        df.to_sql(
            table.name,
            con=con or self.engine,
            dtype={c.name: c.type for c in table.columns},
            if_exists="append",
            index=False,
            method=sqlite_insert_bulk if bulk_insert else None,
        )

    def _store_metadata(
        self,
        metadata_df: pd.DataFrame,
        chunk_ids: pd.Series,
        doc_id: int,
        doc_version: int,
    ):
        with self.engine.begin() as conn:
            key_to_pandas_type = metadata_df.dtypes.to_dict()

            key_to_metadata_types = {
                key: pandas_type_to_metadata_type[pandas_type]
                for key, pandas_type in key_to_pandas_type.items()
                if pandas_type in pandas_type_to_metadata_type
            }
            keys = list(key_to_metadata_types.keys())
            existing_keys = conn.execute(
                select(
                    self.metadata_type_table.c.key, self.metadata_type_table.c.type
                ).where(self.metadata_type_table.c.key.in_(keys))
            ).fetchall()
            existing_key_types = {row.key: row.type for row in existing_keys}

            new_keys = []
            for key, metadata_type in key_to_metadata_types.items():
                existing_type = existing_key_types.get(key)
                if existing_type:
                    if existing_type != metadata_type.value:
                        raise ValueError(
                            f"Type mismatch for key '{key}': existing type '{existing_type}', new type '{metadata_type.value}'"
                        )
                else:
                    new_keys.append({"key": key, "type": metadata_type.value})

            if new_keys:
                conn.execute(self.metadata_type_table.insert(), new_keys)

            metadata_type_to_dfs = defaultdict(list)
            for key, metadata_type in key_to_metadata_types.items():
                metadata_col = metadata_df[key].dropna()
                if metadata_col.empty:
                    continue

                # summarize the metadata
                self.document_metadata_summary.summarize_metadata(
                    key, metadata_col, metadata_type, doc_id, doc_version
                )

                df_to_insert = pd.DataFrame(
                    {
                        "chunk_id": chunk_ids.loc[metadata_col.index],
                        "key": key,
                        "value": metadata_col,
                    }
                )
                metadata_type_to_dfs[metadata_type].append(df_to_insert)

            for metadata_type, dfs in metadata_type_to_dfs.items():
                self._write_to_table(
                    pd.concat(dfs, ignore_index=True),
                    self.metadata_tables[metadata_type],
                    conn,
                    bulk_insert=True,
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

                self._write_to_table(df=chunk_df, table=self.chunk_table)

                if batch.metadata is not None:
                    singlevalue_metadata, multivalue_metadata = (
                        separate_multivalue_columns(batch.metadata)
                    )
                    self._store_metadata(
                        singlevalue_metadata, chunk_ids, doc_id, doc_version
                    )
                    for col in multivalue_metadata:
                        flattened_metadata = flatten_multivalue_column(col, chunk_ids)
                        self._store_metadata(
                            flattened_metadata[[col.name]],
                            flattened_metadata["chunk_id"],
                            doc_id,
                            doc_version,
                        )

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
                    metadata=None,
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

            # TODO(Kartik): The following logic could be handled by a groupby clause
            # and https://www.sqlite.org/lang_aggfunc.html#group_concat
            for row in metadata_results:
                chunk_id = row.chunk_id
                key = row.key
                value = row.value

                if (
                    id_to_chunk[chunk_id].metadata
                    and key in id_to_chunk[chunk_id].metadata
                ):
                    existing_value = id_to_chunk[chunk_id].metadata[key]
                    if isinstance(existing_value, list):
                        existing_value.append(value)
                    else:
                        id_to_chunk[chunk_id].metadata[key] = [existing_value, value]
                else:
                    if id_to_chunk[chunk_id].metadata is None:
                        id_to_chunk[chunk_id].metadata = {}
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

        metadata_types = {}
        missing_columns = []
        with self.engine.begin() as conn:
            is_empty = (
                conn.execute(select(self.metadata_type_table.c.key).limit(1)).fetchone()
                is None
            )

            for column, constraint in constraints.items():
                result = conn.execute(
                    select(self.metadata_type_table.c.type).where(
                        self.metadata_type_table.c.key == column
                    )
                ).fetchone()
                if result:
                    metadata_types[column] = MetadataType(result.type)
                else:
                    missing_columns.append(column)

        if is_empty:
            raise ValueError("Cannot filter constraints with no metadata.")

        if missing_columns:
            raise KeyError(f"Missing columns in metadata: {', '.join(missing_columns)}")

        conditions = []
        query = None
        base_table = None
        for column, constraint in constraints.items():
            metadata_type = metadata_types[column]
            metadata_table = self.metadata_tables[metadata_type]

            metadata_table_alias = alias(metadata_table)

            condition = constraint.sql_condition(
                column_name=column, table=metadata_table_alias
            )
            conditions.append(condition)

            if query is None:
                base_table = metadata_table_alias
                query = select(base_table.c.chunk_id).select_from(base_table)
            else:
                query = query.join(
                    metadata_table_alias,
                    and_(
                        base_table.c.chunk_id == metadata_table_alias.c.chunk_id,
                        metadata_table_alias.c.key == column,
                    ),
                )

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

    def context(self, chunk: Chunk, radius: int) -> List[Chunk]:
        stmt = select(self.chunk_table).where(
            and_(
                self.chunk_table.c.chunk_id >= (chunk.chunk_id - radius),
                self.chunk_table.c.chunk_id <= (chunk.chunk_id + radius),
                self.chunk_table.c.doc_id == chunk.doc_id,
                self.chunk_table.c.doc_version == chunk.doc_version,
            )
        )

        with self.engine.connect() as conn:
            return [
                Chunk(
                    text=row.text,
                    keywords=row.keywords,
                    metadata=None,
                    document=row.document,
                    doc_id=row.doc_id,
                    doc_version=row.doc_version,
                    chunk_id=row.chunk_id,
                )
                for row in conn.execute(stmt).all()
            ]

    def _get_extreme_doc_chunk_ids(self, doc_id: int, doc_version):
        with self.engine.begin() as conn:
            stmt = select(
                func.min(self.chunk_table.c.chunk_id),
                func.max(self.chunk_table.c.chunk_id),
            ).where(
                and_(
                    self.chunk_table.c.doc_id == doc_id,
                    self.chunk_table.c.doc_version == doc_version,
                )
            )
            return conn.execute(stmt).fetchone()

    def _load_summarized_metadata(self, doc_id: int, doc_version: int):
        doc_extreme_chunk_id = self._get_extreme_doc_chunk_ids(doc_id, doc_version)
        if doc_extreme_chunk_id is None:
            raise ValueError("Invalid doc-id or doc-version.")

        with self.engine.begin() as conn:
            min_chunk_id, max_chunk_id = doc_extreme_chunk_id

            document_summarized_metadata = {}
            for metadata_type, metadata_table in self.metadata_tables.items():
                if metadata_type in [MetadataType.FLOAT, MetadataType.INTEGER]:
                    stmt = (
                        select(
                            metadata_table.c.key,
                            func.min(metadata_table.c.value),
                            func.max(metadata_table.c.value),
                        )
                        .where(
                            and_(
                                metadata_table.c.chunk_id >= min_chunk_id,
                                metadata_table.c.chunk_id <= max_chunk_id,
                                metadata_table.c.value.isnot(None),
                            )
                        )
                        .group_by(metadata_table.c.key)
                    )

                    result = conn.execute(stmt).fetchall()
                    for column_name, min_value, max_value in result:
                        document_summarized_metadata[column_name] = (
                            ChunkMetaDataSummary(
                                metadata_type=metadata_type,
                                summary=NumericChunkMetadataSummary(
                                    min=min_value, max=max_value
                                ),
                            )
                        )
                elif metadata_type == MetadataType.STRING:
                    subquery = (
                        select(
                            metadata_table.c.key,
                            metadata_table.c.value,
                            func.row_number()
                            .over(
                                partition_by=metadata_table.c.key,  # Partition by key
                                order_by=func.random(),  # Randomize order within each key
                            )
                            .label("row_number"),  # Assign row numbers
                        )
                        .where(
                            and_(
                                metadata_table.c.chunk_id >= min_chunk_id,
                                metadata_table.c.chunk_id <= max_chunk_id,
                                metadata_table.c.value.isnot(None),
                            )
                        )
                        .group_by(metadata_table.c.key, metadata_table.c.value)
                        .subquery()
                    )
                    stmt = select(subquery.c.key, subquery.c.value).where(
                        subquery.c.row_number <= 100
                    )  # Limit to 100 rows per key

                    result = conn.execute(stmt).fetchall()
                    for column_name, col_value in result:
                        if column_name not in document_summarized_metadata:
                            document_summarized_metadata[column_name] = (
                                ChunkMetaDataSummary(
                                    metadata_type=metadata_type,
                                    summary=StringChunkMetadataSummary(
                                        unique_values=set()
                                    ),
                                )
                            )
                        document_summarized_metadata[
                            column_name
                        ].summary.unique_values.add(col_value)
                else:
                    # Bool metadata type
                    stmt = (
                        select(metadata_table.c.key, metadata_table.c.value)
                        .where(
                            and_(
                                metadata_table.c.chunk_id >= min_chunk_id,
                                metadata_table.c.chunk_id <= max_chunk_id,
                                metadata_table.c.value.isnot(None),
                            )
                        )
                        .group_by(metadata_table.c.key, metadata_table.c.value)
                    )

                    result = conn.execute(stmt).fetchall()
                    for column_name, col_value in result:
                        if column_name not in document_summarized_metadata:
                            document_summarized_metadata[column_name] = (
                                ChunkMetaDataSummary(
                                    metadata_type=metadata_type,
                                    summary=StringChunkMetadataSummary(
                                        unique_values=set()
                                    ),
                                )
                            )
                        document_summarized_metadata[
                            column_name
                        ].summary.unique_values.add(col_value)

        return document_summarized_metadata

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

        if "neural_db_metadata_type" in obj.metadata.tables:
            obj.metadata_tables = {}
            for metadata_type in sql_type_mapping:
                metadata_table_name = f"neural_db_metadata_{metadata_type.value}"
                if metadata_table_name in obj.metadata.tables:
                    obj.metadata_tables[metadata_type] = obj.metadata.tables[
                        metadata_table_name
                    ]

            if "neural_db_metadata_type" in obj.metadata.tables:
                obj.metadata_type_table = obj.metadata.tables["neural_db_metadata_type"]

            obj._create_metadata_tables()
            obj.metadata.create_all(obj.engine)
        else:
            # Migrate deprecated metadata format

            for name in obj.metadata.tables:
                # Few people, if any, have used NDBv2 multivalue metadata columns,
                # so we just throw an error if those tables exist, rather than trying
                # to convert that metadata to the new format.
                if name.startswith("multivalue_metadata_"):
                    raise AttributeError(
                        "Loading NeuralDB with multivalue metadata table is deprecated. Please downgrade thirdai to <= 0.9.20."
                    )

            obj._create_metadata_tables()
            obj.metadata.create_all(obj.engine)

            old_metadata_table = obj.metadata.tables["neural_db_metadata"]

            with obj.engine.connect() as conn:
                old_metadata_df = pd.read_sql(select(old_metadata_table), conn)
                for col in old_metadata_df:
                    if col == "chunk_id":
                        continue
                    obj._store_metadata(
                        old_metadata_df[[col]], old_metadata_df["chunk_id"]
                    )

        with obj.engine.begin() as conn:
            result = conn.execute(select(func.max(obj.chunk_table.c.chunk_id)))
            max_id = result.scalar()
            obj.next_id = (max_id or 0) + 1

        # summarize the metadata
        documents = obj.documents()
        obj.document_metadata_summary = DocumentMetadataSummary()
        for doc_entry in documents:
            doc_id, doc_version = doc_entry["doc_id"], doc_entry["doc_version"]
            obj.document_metadata_summary.summarized_metadata[(doc_id, doc_version)] = (
                obj._load_summarized_metadata(doc_id, doc_version)
            )
        return obj
