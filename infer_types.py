from typing import Any, Dict, List, Tuple

import pandas as pd
import dateutil

TEXT_DELIMETER = ""
CATEGORICAL_DELIMETERS = [";", ":", "-", "\t", "|"]
DELIMETER_RATIO_THRESHOLD = 1.5
DELIMETER_MAP = {
    " ": "text",
    ";": "multi-categorical",
    ":": "multi-categorical",
    "-": "multi-categorical",
    "\t": "multi-categorical",
    "|": "multi-categorical",
}


# Returns the average number of entries per row there would be if the passed in
# delimeter was the actual delimeter
def get_delimeter_ratio(column: pd.Series, delimeter: str) -> float:
    count = sum(column.apply(lambda entry: entry.strip().count(delimeter))) + len(column)
    return count / len(column)

def get_categorical_delimeter_ratios(column: pd.Series) -> Dict[str, float]:
    return {delimeter: get_delimeter_ratio(column, delimeter) for delimeter in CATEGORICAL_DELIMETERS}


def most_occuring_categorical_delimeter(column: pd.Series) -> str:
    ratio_map = get_categorical_delimeter_ratios(column)
    return max(ratio_map, key=lambda entry: ratio_map[entry])


def largest_categorical_delimeter_ratio(column: pd.Series) -> float:
    ratio_map = get_categorical_delimeter_ratios(column)
    return max(ratio_map.values())


def is_delimeted_categorical_col(column: pd.Series) -> bool:
    return largest_categorical_delimeter_ratio(column) > DELIMETER_RATIO_THRESHOLD


def is_text_col(column: pd.Series) -> bool:
    space_ratio = get_delimeter_ratio(column, delimeter=" ")
    return space_ratio > DELIMETER_RATIO_THRESHOLD


def is_datetime_col(column: pd.Series) -> bool:
    try:
        pd.to_datetime(column)
        return True
    except Exception as e:
        return False


def get_col_type(column: pd.Series) -> Dict[str, str]:
    if column.dtype == "float64":
        return {"type": "numerical"}

    if column.dtype == "int64":
        return {"type": "categorical"}

    if column.dtype != "object":
        raise ValueError(
            f"Input columns must be floating point, integers, or text, but found a column of type {column.dtype}"
        )

    if is_datetime_col(column):
        return {"type": "datetime"}

    if is_text_col(column):
        return {"type": "text"}

    if is_delimeted_categorical_col(column):
        return {
            "type": "multi-categorical",
            "delimeter": most_occuring_categorical_delimeter(column),
        }

    return {"type": "categorical"}


def semantic_type_inference(
    filename: str, nrows: int = 100, min_rows_allowed: int = 3
) -> Dict[str, Dict[str, str]]:

    df = pd.read_csv(filename, nrows=nrows)
    if len(df) < min_rows_allowed:
        raise ValueError(
            f"Parsed csv {filename} must have at least {min_rows_allowed} rows, but we found only {len(df)} rows."
        )

    semantic_types = {}
    for column_name in df:

        # Mypy says this can happen sometimes, but I'm not sure when.
        if isinstance(column_name, float):
            raise ValueError(
                f"All columns should have valid names, but found a column with float name {column_name}"
            )

        semantic_types[column_name] = get_col_type(df[column_name])

    return semantic_types


import tempfile


def test_basic_type_inference():
    with tempfile.NamedTemporaryFile(mode="w") as tmp:
        tmp.write(
            """col1,col2,col3,col4,col5,col6
lorem,  2,    3.0,  label1;label2;label3,  How vexingly quick daft zebras jump!,         2021-02-01
ipsum,  5,    6,    label4,                "Sphinx of black quartz judge my vow.",       2022-02-01
dolor,  8,    9,    label5;label6,         The quick brown fox jumps over the lazy dog,  2023-02-01
            """
        )
        tmp.flush()
        
        inferred_types = semantic_type_inference(tmp.name)
        print(inferred_types)

        assert(inferred_types["col1"]["type"] == "categorical")
        assert(inferred_types["col2"]["type"] == "categorical")
        assert(inferred_types["col3"]["type"] == "numerical")
        assert(inferred_types["col4"]["type"] == "multi-categorical")
        assert(inferred_types["col4"]["delimeter"] == ";")
        assert(inferred_types["col5"]["type"] == "text")
        assert(inferred_types["col6"]["type"] == "datetime")


test_basic_type_inference()
