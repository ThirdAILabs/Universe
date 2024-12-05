import itertools
import shutil
import uuid
from collections import defaultdict
from sqlite3 import Connection as SQLite3Connection
from typing import Dict, Iterable, List, Optional, Set, Tuple
import json

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
import subprocess
from .sql_chunk_store import SQLChunkStore

def run_terminal_command(command: list, **kwargs):
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, **kwargs
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error: Command '{' '.join(command)}' failed.")
        print(f"Exit Code: {e.returncode}")
        print(f"Error Output: {e.stderr}")
        raise e
    except FileNotFoundError:
        print(f"Error: Command '{command[0]}' not found. Is it installed?")
        raise


def copy_dataframe_to_postgres(df, table_name, engine, schema=None, sep='\t', null_string=''):
    """
    Efficiently copy a DataFrame to an existing PostgreSQL table using COPY FROM.

    Parameters:
    - df: pandas.DataFrame
        The DataFrame to copy.
    - table_name: str
        The name of the existing table in the database.
    - engine: sqlalchemy.engine.Engine
        The SQLAlchemy engine connected to the database.
    - schema: str, optional
        The schema of the table if not the default (public).
    - sep: str, default '\t'
        Field delimiter for the CSV file.
    - null_string: str, default ''
        The string representation of NULL values.
    """
    # Ensure the table exists
    inspector = inspect(engine)
    if not inspector.has_table(table_name, schema=schema):
        raise ValueError(f"Table '{table_name}' does not exist in the database.")

    # Get the list of columns in the existing table
    db_columns = [col['name'] for col in inspector.get_columns(table_name, schema=schema)]

    # Ensure DataFrame columns match the database table columns
    missing_columns = set(df.columns) - set(db_columns)
    if missing_columns:
        raise ValueError(f"The following columns are missing in the database table: {missing_columns}")

    extra_columns = set(db_columns) - set(df.columns)
    if extra_columns:
        # Optionally handle extra columns in the table that are not in the DataFrame
        # For now, we'll proceed without them
        pass

    # Reorder DataFrame columns to match the database table columns
    df = df[db_columns]

    # Prepare data for COPY FROM
    conn = engine.raw_connection()
    try:
        cur = conn.cursor()
        output = io.StringIO()
        # Use na_rep to represent NaN values as null_string
        df.to_csv(output, sep=sep, header=False, index=False, na_rep=null_string)
        output.seek(0)
        # Prepare the list of columns for the COPY command
        columns_formatted = ', '.join(f'"{col}"' for col in db_columns)
        # Include schema in table name if provided
        if schema:
            table_full_name = f'"{schema}"."{table_name}"'
        else:
            table_full_name = f'"{table_name}"'
        # COPY command with column names
        copy_sql = f"COPY {table_full_name} ({columns_formatted}) FROM STDIN WITH CSV DELIMITER '{sep}' NULL '{null_string}'"
        cur.copy_expert(copy_sql, output)
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Error during COPY FROM operation: {e}")
        raise
    finally:
        cur.close()
        conn.close()

class PostgreSQLChunkStore(SQLChunkStore):
    def __init__(
        self,
        postgresql_admin_uri: str,
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
            postgresql_uri: Optional[str] - If provided, the chunk store will use postgres rather than sqlite
        """

        random_string = ''.join(random.choices(string.ascii_lowercase, k=16))
        chunk_store_db_name = f"ndb_{random_string}"
        self.postgresql_admin_uri = postgresql_admin_uri

        sql_uri = self.create_psql_database(chunk_store_db_name)

        on_disk_db_name = save_path or f"{chunk_store_db_name}.json"
        data = {"sql_uri": sql_uri}
        with open(on_disk_db_name, "w") as file:
            json.dump(data, file, indent=4)
        
        super().__init__(encryption_key=encryption_key, use_metadata_index=use_metadata_index, sql_uri=sql_uri, **kwargs)

    def create_psql_database(self, db_name):
        engine = create_engine(self.postgresql_admin_uri)
        try:
            with engine.connect() as connection:
                connection.execution_options(isolation_level="AUTOCOMMIT")
                connection.execute(text(f"CREATE DATABASE {db_name}"))

            sql_uri = f"{self.postgresql_admin_uri.rsplit('/', 1)[0]}/{db_name}"
            return sql_uri
        except ProgrammingError as e:
            print(f"An error occurred in creating database '{db_name}': {e}")
            raise e


    def _write_to_table(
        self, df: pd.DataFrame, table: Table, con=None,
    ):
        copy_dataframe_to_postgres(
            df,
            table_name=table.name,
            engine=self.engine,
            sep='\t',
            null_string=''
        )

    def copy_database(self, target_uri):
        """Copy the contents of one PostgreSQL database to another."""
        try:
            dump_command = [
                "pg_dump",
                self.sql_uri,
                "--clean",
                "--no-owner",
            ]
            dump_output = run_terminal_command(dump_command)

            restore_command = [
                "psql",
                target_uri,
            ]
            run_terminal_command(restore_command, input=dump_output)
            print(f"Database '{self.sql_uri}' successfully copied to '{target_uri}'.")
        except Exception as e:
            print(f"Failed to copy database: {e}")
            raise

    def save(self, path: str):
        random_string = ''.join(random.choices(string.ascii_lowercase, k=16))
        new_chunk_store_db_name = f"ndb_{random_string}"
        target_uri = self.create_psql_database(new_chunk_store_db_name)
        self.copy_database(target_uri)
        data = {"sql_uri": target_uri}
        with open(path, "w") as file:
            json.dump(data, file, indent=4)

    def _get_sql_uri(self, path):
        with open(path, "r") as file:
            data = json.load(file)

        self.postgresql_admin_uri = data["sql_uri"]
        return data["sql_uri"]