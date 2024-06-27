import typing
from dataclasses import dataclass

import pandas as pd
from thirdai import bolt

import utils


@dataclass
class Column:
    column_name: str

    def to_bolt(self):
        raise NotImplementedError()


@dataclass
class CategoricalColumn(Column):
    column_name: str
    token_type: str
    number_tokens_per_row: float
    unique_tokens_per_row: float
    estimated_n_classes: int

    delimiter: typing.Optional[str] = None

    def to_bolt(self, is_target_type=False):
        return bolt.types.categorical(
            type=self.token_type,
            delimiter=self.delimiter,
            n_classes=self.estimated_n_classes if is_target_type else None,
        )


@dataclass
class NumericalColumn(Column):
    column_name: str
    minimum: float
    maximum: float

    def to_bolt(self):
        return bolt.types.numerical((self.minimum, self.maximum))


@dataclass
class DateTimeColumn(Column):
    column_name: str

    def to_bolt(self):
        return bolt.types.date()


@dataclass
class TextColumn(Column):
    column_name: str

    def to_bolt(self):
        return bolt.types.text()


@dataclass
class TokenTags(Column):
    default_tag: str
    named_tags: typing.List[str]

    def to_bolt(self):
        return bolt.types.token_tags(tags=self.named_tags, default_tag=self.default_tag)


@dataclass
class SequenceType(Column):
    column_name: str
    delimiter: str
    estimated_n_classes: int = None
    max_length: int = None

    def __post_init__(self):
        if self.delimiter is None:
            raise Exception(
                "The delimiter for a sequence type column is None. Ensure that the entries in a column are valid sequences."
            )

    def to_bolt(self):
        return bolt.types.sequence(
            delimiter=self.delimiter,
            n_classes=self.estimated_n_classes,
            max_length=self.max_length,
        )


def cast_to_categorical(column_name: str, column: pd.Series):
    delimiter, token_row_ratio = utils.find_delimiter(column)
    unique_values_in_column = utils.get_unique_values(column, delimiter)
    token_data_type = utils.get_numerical_data_type(unique_values_in_column)

    utils.cast_set_values(unique_values_in_column, token_data_type)

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
    
    if (
        categorical_column.delimiter == None
        and categorical_column.token_type != 'str'
    ):
        if categorical_column.token_type == "float":
            return cast_to_numerical(column_name, dataframe[column_name])

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
) -> TextColumn:

    if target.delimiter != " ":
        raise Exception("Expected the target column to be space seperated tags.")

    candidate_columns: typing.List[Column] = []
    for column_name, column in input_columns.items():
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

    if target.delimiter != " ":
        raise Exception("Expected the target column to be space seperated tags.")

    candidate_columns: typing.List[CategoricalColumn] = []
    for column_name, column in input_columns.items():
        if isinstance(column, CategoricalColumn):
            if column.delimiter == " ":
                ratio_source_to_target = (
                    column.number_tokens_per_row / target.number_tokens_per_row
                )
                if ratio_source_to_target > 1.5 or ratio_source_to_target < 0.66:
                    continue

                candidate_columns.append(TextColumn(column_name=column.column_name))

    return candidate_columns
