from core.index import Index
from indexes.dataframe_index import DataFrameIndex


def index_by_name(name: str, **kwargs):
    if name == "default":
        return DataFrameIndex(**kwargs)
    if name == "dataframe":
        return DataFrameIndex(**kwargs)
    if name == "inmemory":
        return DataFrameIndex(**kwargs)
    raise ValueError(f"Invalid index name {name}")
