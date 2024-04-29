from pathlib import Path

from pandas_chunk_store import PandasChunkStore
from sqlite_chunk_store import SQLiteChunkStore

chunk_store_name_map = {
    "PandasChunkStore": PandasChunkStore,
    "SQLiteChunkStore": SQLiteChunkStore,
}


def load_chunk_store(path: Path, chunk_store_name: str):
    if chunk_store_name not in chunk_store_name_map:
        raise ValueError(f"Class name {chunk_store_name} not found in registry.")

    return chunk_store_name_map[chunk_store_name].load(path)
