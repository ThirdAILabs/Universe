import typing
from collections import defaultdict

import pandas as pd

from . import utils
from .columns import *


def cast_to_categorical(
    column_name: str, column: pd.Series, cast_float_to_str: bool = True
):
    delimiter, token_row_ratio = utils.find_delimiter(column)
    unique_values_in_column = utils.get_unique_values(column, delimiter)
    token_data_type = utils.get_token_data_type(unique_values_in_column)

    unique_values_in_column = {token_data_type(val) for val in unique_values_in_column}

    # For categorical columns, we can only get an estimate of the number of
    # unique tokens since the user specified dataframe might not be comprehensive
    # representation of the entire dataset.
    if token_data_type == str or token_data_type == float:
        n_classes = len(unique_values_in_column)
    else:
        n_classes = max(unique_values_in_column) + 1

    if token_data_type == float and cast_float_to_str:
        token_data_type = str

    return CategoricalColumn(
        name=column_name,
        token_type=token_data_type,
        number_tokens_per_row=token_row_ratio,
        unique_tokens_per_row=len(unique_values_in_column) / len(column),
        delimiter=delimiter,
        estimated_n_classes=n_classes,
    )


def cast_to_numerical(column_name: str, column: pd.Series):
    try:
        column = pd.to_numeric(column)
        return NumericalColumn(
            name=column_name, minimum=column.min(), maximum=column.max()
        )
    except:
        return None


def detect_single_column_type(
    column_name, dataframe: pd.DataFrame
) -> typing.Union[DateTimeColumn, NumericalColumn, CategoricalColumn]:
    """
    Identifies and classifies the type of a given column in a DataFrame into one of three distinct categories:
    DateTime, Numerical, or Categorical.
    """

    if utils._is_datetime_col(dataframe[column_name]):
        return DateTimeColumn(name=column_name)

    categorical_column = cast_to_categorical(
        column_name, dataframe[column_name], cast_float_to_str=False
    )

    if categorical_column.delimiter == None and categorical_column.token_type != str:
        if categorical_column.token_type == float:
            numerical_column = cast_to_numerical(column_name, dataframe[column_name])
            if numerical_column:
                return numerical_column

        if (
            categorical_column.token_type == int
            and categorical_column.estimated_n_classes > 100_000
        ):
            # If the number of unique tokens in the column is high, then chances are that it is numerical.
            return cast_to_numerical(column_name, dataframe[column_name])

    # the below condition means that there is a delimiter in the column. A column with multiple floats
    # in a single row will be treated as a string multicategorical column
    if categorical_column.token_type == float:
        categorical_column.token_type = str

    return categorical_column


def get_input_columns(
    target_column_name, dataframe: pd.DataFrame
) -> typing.Dict[str, Column]:
    """
    Extracts the data types of columns in a given pandas DataFrame, excluding the specified target column.
    """

    input_data_types = {}

    for col in dataframe.columns:
        if col == target_column_name:
            continue

        input_data_types[col] = detect_single_column_type(
            column_name=col, dataframe=dataframe
        )

    return input_data_types


def get_token_candidates_for_token_classification(
    target: CategoricalColumn, input_columns: typing.Dict[str, Column]
) -> typing.List[TextColumn]:
    """
    Identifies candidate columns suitable as token columns for a TokenClassification task
    where each token must have a corresponding tag.

    Returns a list of candidates suitable to be the tokens column for the task.
    """

    if target.delimiter != " " and target.token_type != str:
        raise Exception("Expected the target column to be space seperated tags.")

    candidate_columns: typing.List[Column] = []
    for _, column in input_columns.items():
        if isinstance(column, CategoricalColumn):
            if (
                column.delimiter == " "
                and (column.number_tokens_per_row == target.number_tokens_per_row)
                and column.token_type == str
            ):
                candidate_columns.append(TextColumn(name=column.name))
    return candidate_columns


def get_source_column_for_query_reformulation(
    target: CategoricalColumn, input_columns: typing.Dict[str, Column]
) -> TextColumn:
    """
    Returns a list of columns where each column is a candidate to be the source column for the specified target column.
    """

    if target.delimiter != " " and target.token_type != str:
        raise Exception("Expected the target column to be space seperated tokens.")

    candidate_columns: typing.List[CategoricalColumn] = []
    for _, column in input_columns.items():
        if isinstance(column, CategoricalColumn):
            if column.delimiter == " ":
                ratio_source_to_target = (
                    column.number_tokens_per_row / target.number_tokens_per_row
                )
                if (
                    ratio_source_to_target > 1.5
                    or ratio_source_to_target < 0.66
                    or column.token_type != str
                ):
                    continue

                candidate_columns.append(TextColumn(name=column.name))

    return candidate_columns


def get_frequency_sorted_unique_tokens(
    target: CategoricalColumn, dataframe: pd.DataFrame
):
    tag_frequency_map = defaultdict(int)

    def add_key_to_dict(dc, key):
        if key.strip():
            dc[key.strip()] += 1

    dataframe[target.name].apply(
        lambda row: [add_key_to_dict(tag_frequency_map, key) for key in row.split(" ")]
    )

    sorted_tags = sorted(
        tag_frequency_map.items(), key=lambda item: item[1], reverse=True
    )
    sorted_tag_keys = [tag for tag, freq in sorted_tags]

    return sorted_tag_keys
