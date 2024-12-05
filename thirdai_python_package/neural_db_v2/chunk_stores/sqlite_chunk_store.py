import itertools
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
    text,
    Text,
    cast,
    inspect
)
from sqlalchemy.engine import Engine
from sqlalchemy_utils import StringEncryptedType
from sqlalchemy_utils.types.encrypted.encrypted_type import AesEngine
from sqlalchemy.exc import ProgrammingError

from ..core.chunk_store import ChunkStore
from ..core.documents import Document
from ..core.types import (
    Chunk,
    ChunkBatch,
    ChunkId,
    InsertedDocMetadata,
    MetadataType,
    pandas_type_to_metadata_type,
    sql_type_mapping,
)
from .constraints import Constraint
import random
import string

import csv
from io import StringIO
import io
import psycopg2
from .sql_chunk_store import SQLChunkStore


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

class SQLiteChunkStore(SQLChunkStore):
    def __init__(
        self,
        save_path: Optional[str] = None,
        encryption_key: Optional[str] = None,
        use_metadata_index: bool = False,
        postgresql_uri: Optional[str] = None,
        **kwargs,
    ):
        """
        Params:
            save_path: Optional[str] - Path to save db to, otherwise is random
            encryption_key: Optional[str] - Must be passed to encrypt data
            use_metadata_index: bool - If true, insertion time doubles but query time with constraints roughly halves
            postgresql_uri: Optional[str] - If provided, the chunk store will use postgres rather than sqlite
        """
        
        on_disk_db_name = save_path or f"{uuid.uuid4()}.db"
        sql_uri = f"sqlite:///{on_disk_db_name}"
        super().__init__(encryption_key=encryption_key, use_metadata_index=use_metadata_index, sql_uri=sql_uri, **kwargs)

    def _write_to_table(
        self, df: pd.DataFrame, table: Table, con=None
    ):
        df.to_sql(
            table.name,
            con=con or self.engine,
            dtype={c.name: c.type for c in table.columns},
            if_exists="append",
            index=False,
            method=sqlite_insert_bulk,
        )

    def save(self, path: str):
        shutil.copyfile(self.sql_uri, path)

    def _get_sql_uri(self, path):
        return f"sqlite:///{path}"