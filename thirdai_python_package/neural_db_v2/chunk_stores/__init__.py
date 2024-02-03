from chunk_stores.dataframe_index import DataFrameIndex

from thirdai_python_package.neural_db_v2.core.chunk_store import ChunkStore


def chunk_store_by_name(name: str, **kwargs) -> ChunkStore:
    if name == "default":
        return DataFrameIndex(**kwargs)
    if name == "dataframe":
        return DataFrameIndex(**kwargs)
    if name == "inmemory":
        return DataFrameIndex(**kwargs)
    # TODO: Add more options, e.g. SQLite
    raise ValueError(f"Invalid index name {name}")
