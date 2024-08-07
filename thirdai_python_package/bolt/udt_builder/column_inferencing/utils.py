import warnings
from typing import Optional, Tuple, Union

warnings.filterwarnings("ignore")
import pandas as pd

candidate_delimiters = [";", ":", "-", "\t", "|", " "]


def _is_datetime_col(column: pd.Series) -> bool:
    try:
        pd.to_numeric(column)
        return False
    except:
        try:
            pd.to_datetime(column)
            return True
        except Exception as e:
            return False


def is_int(val: str) -> bool:
    """
    Checks if string can be cast to a valid int.
    """
    try:
        casted_value = int(val)
        return True
    except:
        return False


def is_float(val: str) -> bool:
    """
    Checks if string can be cast to a valid float.
    """
    try:
        casted_value = float(val)
        return True
    except:
        return False


def get_token_data_type(values: set) -> Union[str, int, float]:
    """
    Finds the most suitable numerical data type for a set of string values:
    returns 'int' if all can be integers, 'float' if all are floats, otherwise 'str'.
    """
    numerical_data_type = int
    for val in values:
        if not is_int(val):
            numerical_data_type = float
            if not is_float(val):
                return str

    return numerical_data_type


def _get_delimiter_ratio(column: pd.Series, delimiter: str) -> float:
    count = sum(column.apply(lambda entry: entry.strip().count(delimiter))) + len(
        column
    )
    return count / len(column)


def get_unique_values(column: pd.Series, delimiter: str) -> float:
    unique_values = set()
    column.apply(
        lambda row: (
            unique_values.update(row.strip().split(delimiter))
            if delimiter
            else unique_values.add(row)
        )
    )
    return unique_values


def find_delimiter(column: pd.Series) -> Tuple[Optional[str], float]:
    """
    Returns the most frequent delimiter in the series.
    """
    ratios = {
        delimiter: _get_delimiter_ratio(column, delimiter)
        for delimiter in candidate_delimiters
    }
    most_occurring_delimiter = max(ratios, key=lambda entry: ratios[entry])
    most_occurring_ratio = ratios[most_occurring_delimiter]

    if most_occurring_ratio > 1:
        return most_occurring_delimiter, most_occurring_ratio

    return None, most_occurring_ratio
