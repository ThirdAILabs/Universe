from thirdai_python_package.neural_db_v2.chunk_stores.dataframe_chunk_store import (
    DataFrameChunkStore,
)

from thirdai_python_package.neural_db_v2.core.chunk_store import ChunkStore


def chunk_store_by_name(name: str, **kwargs) -> ChunkStore:
    if name == "default":
        return DataFrameChunkStore(**kwargs)
    if name == "dataframe":
        return DataFrameChunkStore(**kwargs)
    if name == "inmemory":
        return DataFrameChunkStore(**kwargs)
    # TODO: Add more options, e.g. SQLite
    raise ValueError(f"Invalid index name {name}")
