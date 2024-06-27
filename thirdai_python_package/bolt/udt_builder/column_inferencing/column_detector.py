import typing
import pandas as pd
from collections import defaultdict

from . import utils
from .columns import *


def cast_to_categorical(column_name: str, column: pd.Series):
    delimiter, token_row_ratio = utils.find_delimiter(column)
    unique_values_in_column = utils.get_unique_values(column, delimiter)
    token_data_type = utils.get_numerical_data_type(unique_values_in_column)

    utils.cast_set_values(unique_values_in_column, token_data_type)

    # For categorical columns, we can only get an estimate of the number of
    # unique tokens since the user specified dataframe might not be comprehensive
    # representation of the entire dataset.
    if token_data_type == "str" or token_data_type == "float":
        n_classes = len(unique_values_in_column)
        token_data_type = "str"
    else:
        n_classes = max(unique_values_in_column) + 1

    return CategoricalColumn(
        column_name=column_name,
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
            column_name=column_name, minimum=min(column), maximum=max(column)
        )
    except:
        return None


def detect_single_column_type(column_name, dataframe: pd.DataFrame):
    """
    We segment the column into 3 seperate types which are fairly dissimilar to one another :
    1. DateTime Column
    2. Numerical Column
    3. Categorical Column

    We use a bunch of heuristics to identify an Int Numerical Column from another Int Single Categorical Column. If the number of unique tokens in the column is high, then chances are that it is numerical.
    """
    if utils._is_datetime_col(dataframe[column_name]):
        return DateTimeColumn(column_name=column_name)

    categorical_column = cast_to_categorical(column_name, dataframe[column_name])

    if categorical_column.delimiter == None and categorical_column.token_type != "str":
        if categorical_column.token_type == "float":
            return cast_to_numerical(column_name, dataframe[column_name])

        if (
            categorical_column.token_type == "int"
            and categorical_column.estimated_n_classes > 100_000
        ):
            return cast_to_numerical(column_name, dataframe[column_name])

    """
    uncomment this block and replace the previous if block to make numerical features for a column
    if the ratio of unique integers exceeds a certain threshold.
    
        if categorical_column.token_type == "int" and (
            categorical_column.unique_tokens_per_row > 0.6
            or categorical_column.estimated_n_classes > 100_000
        ):
            return cast_to_numerical(column_name, dataframe[column_name])
    """

    # the below condition means that there is a delimiter in the column. A column with multiple floats
    # in a single row will be treated as a string multicategorical column
    if categorical_column.token_type == "float":
        categorical_column = "str"

    return categorical_column


def get_input_columns(target_column_name, dataframe: pd.DataFrame) -> typing.Dict:
    input_data_types = {}

    for col in dataframe.columns:
        if col == target_column_name:
            continue

        input_data_types[col] = detect_single_column_type(
            column_name=col, dataframe=dataframe
        )

    return input_data_types


def get_token_candidates_for_token_classification(
    target: CategoricalColumn,
    input_columns: typing.Dict[str, Column],
) -> typing.List[TextColumn]:
    """
    Returns a list of columns where each column is a candidate to be the token column for the specified target column (assuming the task is TokenClassification).
    """

    if target.delimiter != " ":
        raise Exception("Expected the target column to be space seperated tags.")

    candidate_columns: typing.List[Column] = []
    for _, column in input_columns.items():
        if isinstance(column, CategoricalColumn):
            if (
                column.delimiter == " "
                and abs(column.number_tokens_per_row - target.number_tokens_per_row)
                < 0.001
            ):
                candidate_columns.append(TextColumn(column_name=column.column_name))
    return candidate_columns


def get_source_column_for_query_reformulation(
    target: CategoricalColumn,
    input_columns: typing.Dict[str, Column],
) -> TextColumn:
    """
    Returns a list of columns where each column is a candidate to be the source column for the specified target column (assuming the task is QueryClassification).
    """

    if target.delimiter != " " and target.token_type != "str":
        raise Exception("Expected the target column to be space seperated tokens.")

    candidate_columns: typing.List[CategoricalColumn] = []
    for _, column in input_columns.items():
        if isinstance(column, CategoricalColumn):
            if column.delimiter == " ":
                ratio_source_to_target = (
                    column.number_tokens_per_row / target.number_tokens_per_row
                )
                if ratio_source_to_target > 1.5 or ratio_source_to_target < 0.66:
                    continue

                candidate_columns.append(TextColumn(column_name=column.column_name))

    return candidate_columns


def get_frequency_sorted_unique_tokens(
    target: CategoricalColumn, dataframe: pd.DataFrame
):
    tag_frequency_map = defaultdict(int)

    def add_key_to_dict(dc, key):
        if key.strip():
            dc[key] += 1

    dataframe[target.column_name].apply(
        lambda row: [add_key_to_dict(tag_frequency_map, key) for key in row.split(" ")]
    )

    sorted_tags = sorted(
        tag_frequency_map.items(), key=lambda item: item[1], reverse=True
    )
    sorted_tag_keys = [tag for tag, freq in sorted_tags]

    return sorted_tag_keys
