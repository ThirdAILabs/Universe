from typing import Any, Dict, List, Tuple

import pandas as pd

TEXT_DELIMETER = ""
CATEGORICAL_DELIMETERS = [";", ":", "-", "\t", "|"]
DELIMETER_RATIO_THRESHOLD = 1
DELIMETER_MAP = {
    " ": "text",
    ";": "multi-categorical",
    ":": "multi-categorical",
    "-": "multi-categorical",
    "\t": "multi-categorical",
    "|": "multi-categorical",
}


def get_delimeter_ratio(column: pd.Series, delimeter: str) -> float:
    count = sum(column.apply(lambda entry: entry.count(delimeter)))
    return count / len(column)


def most_occuring_categorical_delimeter_and_count(column: pd.Series) -> Tuple[str, int]:
    most_occuring_delimeter = CATEGORICAL_DELIMETERS[0]
    most_occuring_delimeter_count = -1
    for delimeter in CATEGORICAL_DELIMETERS:
        count = sum(column.apply(lambda entry: entry.count(delimeter)))
        if count > most_occuring_categorical_delimeter_and_count:
            most_occuring_delimeter = delimeter
            most_occuring_delimeter_count = count

    return most_occuring_delimeter, most_occuring_delimeter_count


def most_occuring_categorical_delimeter(column: pd.Series) -> str:
    return most_occuring_categorical_delimeter_and_count(column)[0]


def most_occuring_categorical_delimeter_ratio(column: pd.Series) -> float:
    return most_occuring_categorical_delimeter_and_count(column)[1] / len(column)


def is_delimeted_categorical_col(column: pd.Series) -> bool:
    return most_occuring_categorical_delimeter_ratio(column) > DELIMETER_RATIO_THRESHOLD


def is_text_col(column: pd.Series) -> bool:
    space_ratio = get_delimeter_ratio(column, delimeter=" ")
    return space_ratio > DELIMETER_RATIO_THRESHOLD


def is_datetime_col(column: pd.Series) -> bool:
    try:
        pd.to_datetime(column)
        return True
    except pd.errors.ParserError:
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
            "deliemter": most_occuring_categorical_delimeter(column),
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


def test_type_inference():
    with tempfile.NamedTemporaryFile(mode="w") as tmp:
        tmp.write(
            """col1,col2,col3
            1,2,3.0
            4,5,6
            7,8,9
            """
        )
        tmp.flush()
        
        inferred_types = semantic_type_inference(tmp.name)
        print(inferred_types)
        assert(inferred_types["col1"]["type"] == "categorical")
        assert(inferred_types["col2"]["type"] == "categorical")
        assert(inferred_types["col3"]["type"] == "numerical")


test_type_inference()
