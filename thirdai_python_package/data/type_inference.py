from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

_TEXT_DELIMITER = " "
_CATEGORICAL_DELIMITERS = [";", ":", "-", "\t", "|"]
# If the most occurring delimiter would cause there to be more than _DELIMITER_RATIO_THRESHOLD
# items per line, that is enough to assume that these are valid splits and that
# the column is indeed multi-categorical (or text)
_DELIMITER_RATIO_THRESHOLD = 1.5


# Returns the average number of entries per row there would be if the passed in
# delimiter was the actual delimiter
def _get_delimiter_ratio(column: pd.Series, delimiter: str) -> float:
    count = sum(column.apply(lambda entry: entry.strip().count(delimiter))) + len(
        column
    )
    return count / len(column)


# Returns a tuple where the first item is whether the column is multi-categorical
# and the second is the column's delimiter if it is multi-categorical and None
# otherwise.
def _is_multi_categorical(column: pd.Series) -> Tuple[bool, Optional[str]]:
    ratios = {
        delimiter: _get_delimiter_ratio(column, delimiter)
        for delimiter in _CATEGORICAL_DELIMITERS
    }

    most_occurring_delimiter = max(ratios, key=lambda entry: ratios[entry])
    most_occurring_ratio = ratios[most_occurring_delimiter]

    if most_occurring_ratio >= _DELIMITER_RATIO_THRESHOLD:
        return True, most_occurring_delimiter

    return False, None


def _is_text_col(column: pd.Series) -> bool:
    space_ratio = _get_delimiter_ratio(column, delimiter=_TEXT_DELIMITER)
    return space_ratio > _DELIMITER_RATIO_THRESHOLD


def _is_datetime_col(column: pd.Series) -> bool:
    try:
        pd.to_datetime(column)
        return True
    except Exception as e:
        return False


def _is_float_col(column: pd.Series):
    try:
        converted_col = pd.to_numeric(column)
        return converted_col.dtype == "float64"
    except:
        return False


def _is_int_col(column: pd.Series):
    try:
        converted_col = pd.to_numeric(column)
        return converted_col.dtype == "int64"
    except:
        return False


def _infer_col_type(column: pd.Series) -> Dict[str, str]:
    # Since Pandas reads blank values as NA in read_csv, this will drop missing
    # values, thus allowing us to do type inference correctly.
    column = column.dropna()

    if len(column) < 2:
        raise ValueError(
            f"Column {column} has less than 2 non-missing values so we cannot do type inference"
        )

    if _is_float_col(column):
        return {"type": "numerical"}

    if _is_int_col(column):
        return {"type": "categorical"}

    if _is_datetime_col(column):
        return {"type": "datetime"}

    if _is_text_col(column):
        return {"type": "text"}

    is_multi_categorical, delimiter = _is_multi_categorical(column)
    if is_multi_categorical:
        assert delimiter is not None
        return {
            "type": "multi-categorical",
            "delimiter": delimiter,
        }

    return {"type": "categorical"}


def semantic_type_inference(
    filename: str, nrows: int = 100, min_rows_allowed: int = 3
) -> Dict[str, Dict[str, str]]:
    """Tries to parse the given filename as a csv and then infer the type of each
    column.

    Args:
        filename: The filename to use for type inference. The file should be
            a CSV with no extra whitespace between items.

        nrows: The number of rows of the file to use to infer column types. If
            the file has less than nrows number of rows, all rows will be used.

        min_rows_allowed: The minimum number of rows we allow for inferring
            types. If we find less than min_rows_allowed rows, we will throw a
            ValueError.

    Returns:
        A map from column name to a metadata dictionary. One of the items in the
        metadata dictionary will be {"type": X}, where X is one of ["categorical",
        "multi-categorical", "datetime", "numerical", "text"]. Types that are
        "multi-categorical" will have an additional item in the dictionary
        specifying the estimate delimiter. Refer to the infer_types_test.py files
        for full examples with expected inputs and outputs.
    """
    try:
        if filename.endswith(".pqt") or filename.endswith(".parquet"):
            df = pd.read_parquet(filename)
        else:
            # We force dtype=object so that int and string columns are treated correctly
            # even with missing values (we will later drop the missing values, which
            # get converted to NAs during read_csv, and then convert to the correct
            # more specific type).
            df = pd.read_csv(filename, nrows=nrows, dtype=object)
    except:
        raise ValueError(
            "UDT currently supports all files that can be read using "
            "pandas.read_parquet (for .pqt or .parquet files) or "
            "pandas.read_csv (for all other files). Please convert your files "
            "to one of the supported formats."
        )

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

        semantic_types[column_name] = _infer_col_type(df[column_name])

    return semantic_types
