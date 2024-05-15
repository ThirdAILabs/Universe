from typing import Any, List

import pandas as pd
from pandas.api import types as pd_types
from sqlalchemy import (
    Column,
    Engine,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    select,
)


def get_engine(db_path: str):
    return create_engine(f"sqlite:///{db_path}")


def infer_type(series: pd.Series):
    if pd_types.is_integer_dtype(series):
        return Integer()
    if pd_types.is_float_dtype(series):
        return Float()
    return String()


def infer_types(df: pd.DataFrame):
    types = {col: infer_type(df[col]) for col in df.columns}
    types[df.index.name] = Integer()
    return types


def create_table(engine: Engine, df: pd.DataFrame, name: str, types: dict):
    metadata_obj = MetaData()
    columns = [
        Column(col, dtype, primary_key=col == df.index.name)
        for col, dtype in types.items()
    ]
    table = Table(
        name,
        metadata_obj,
        *columns,
    )
    metadata_obj.create_all(engine)
    return table


def df_to_sql(db_path: str, df: pd.DataFrame, table_name: str):
    engine = get_engine(db_path)
    types = infer_types(df)
    table = create_table(engine, df, table_name, types)
    df.to_sql(
        name=table_name,
        con=engine,
        index=True,
        dtype=types,
        if_exists="append",
    )
    return table


def select_as_df(db_path: str, table: Table, constraints: List[Any] = None):
    engine = get_engine(db_path)
    selection = select(table)
    if constraints:
        for constraint in constraints:
            selection = selection.where(constraint)
    return pd.read_sql(selection, engine)
