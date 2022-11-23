from typing import Dict

import pandas as pd
import thirdai._thirdai.bolt as bolt

from .type_inference import semantic_type_inference


def get_udt_col_types(
    filename: str, n_rows: int = 1e6
) -> Dict[str, bolt.types.ColumnType]:
    """Returns a best guess for the types and metadata of the columns of the
            input file.

    Args:
        filename (str): Path to a csv stored locally or on aws/gcp/etc (anything
            that can be read by pandas.read_csv).

    Returns:
        (Dict[str, bolt.types.ColumnType]):
        A map from column name to our best guess for ColumnType.
    """
    column_types = semantic_type_inference(filename)

    df = pd.read_csv(filename, nrows=n_rows, low_memory=False)

    udt_column_types = {}

    for col_name, type_info in column_types.items():
        if col_name not in df.columns:
            raise ValueError(
                f"column_type map contains column: {col_name} not in dataframe."
            )
        col_type = type_info["type"]

        if col_type == "text":
            udt_column_types[col_name] = bolt.types.text()
        elif col_type == "categorical":
            udt_column_types[col_name] = bolt.types.categorical()
        elif col_type == "multi-categorical":
            udt_column_types[col_name] = bolt.types.categorical(
                delimiter=type_info["delimiter"]
            )
        elif col_type == "numerical":
            min_val = df[col_name].min()
            max_val = df[col_name].max()
            udt_column_types[col_name] = bolt.types.numerical(range=(min_val, max_val))
        elif col_type == "datetime":
            udt_column_types[col_name] = bolt.types.date()
        else:
            raise ValueError(
                f"Received invalid column type: {col_type}. Supports 'text', 'categorical', 'multi-categorical', 'numerical', and 'datetime'."
            )

    return udt_column_types
