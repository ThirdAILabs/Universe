import pandas as pd
import pytest
from thirdai.bolt.udt_modifications import task_detector

from udt_builder_utils import *

pytestmark = [pytest.mark.unit, pytest.mark.release]


@pytest.fixture(scope="module")
def ner_dataframe():
    dataframe = pd.DataFrame(
        {
            "float_col": get_numerical_column(NUM_ROWS, 0, 100, create_floats=True),
            "int_col": get_numerical_column(NUM_ROWS, 0, 100, create_floats=False),
            "int_categorical_col": get_int_categorical_column(100, 5, 0, 1000, ":"),
            "string_categorical_col": get_string_categorical_column(100, 5, " "),
            "ner_target_col": get_string_categorical_column(
                100, 5, " ", select_tokens_from=["O", "A", "B"]
            ),
            "query_reformulation_target": get_string_categorical_column(100, 6, " "),
        }
    )
    return dataframe


@pytest.mark.parametrize(
    "number_tokens_per_row,delimiter", [(1, " "), (5, " "), (6, ":"), (7, "-")]
)
def test_auto_infer_string_categorical_column(number_tokens_per_row, delimiter):
    col = get_string_categorical_column(100, number_tokens_per_row, delimiter)
    df = pd.DataFrame({"col": col})
    column = task_detector.column_detector.detect_single_column_type("col", df)

    assert isinstance(column, task_detector.column_detector.CategoricalColumn)
    assert column.token_type == str
    assert column.number_tokens_per_row == number_tokens_per_row

    if number_tokens_per_row == 1:
        assert column.delimiter == None
    else:
        assert column.delimiter == delimiter


@pytest.mark.parametrize(
    "number_tokens_per_row,delimiter", [(1, " "), (5, " "), (6, ":"), (7, "-")]
)
def test_auto_infer_int_categorical_column(number_tokens_per_row, delimiter):
    col = get_int_categorical_column(100, number_tokens_per_row, 0, 10000, delimiter)
    df = pd.DataFrame({"col": col})
    column = task_detector.column_detector.detect_single_column_type("col", df)

    assert isinstance(column, task_detector.column_detector.CategoricalColumn)
    assert column.name == "col"
    assert column.token_type == int
    assert column.number_tokens_per_row == number_tokens_per_row

    if number_tokens_per_row == 1:
        assert column.delimiter == None
    else:
        assert column.delimiter == delimiter


@pytest.mark.parametrize("create_floats", [True, False])
def test_auto_infer_numerical_column(create_floats):
    col = get_numerical_column(NUM_ROWS, 0, 10_000_000, create_floats=create_floats)

    df = pd.DataFrame({"col": col})
    column = task_detector.column_detector.detect_single_column_type("col", df)

    assert isinstance(column, task_detector.column_detector.NumericalColumn)
    assert column.name == "col"
    assert column.maximum < 10_000_000
    assert column.minimum >= 0


def test_auto_infer_token_classification_candidates(ner_dataframe):
    target_column = task_detector.column_detector.detect_single_column_type(
        "ner_target_col", ner_dataframe
    )
    input_columns = task_detector.column_detector.get_input_columns(
        "ner_target_col", ner_dataframe
    )

    ner_source_candidates = (
        task_detector.column_detector.get_token_candidates_for_token_classification(
            target_column, input_columns
        )
    )

    assert len(ner_source_candidates) == 1
    assert ner_source_candidates[0].name == "string_categorical_col"

    detected_tags = set(
        task_detector.column_detector.get_frequency_sorted_unique_tokens(
            target_column, ner_dataframe
        )
    )

    assert len(detected_tags) == 3
    assert set(["O", "A", "B"]) == detected_tags


def test_auto_infer_query_reformulation_candidates(ner_dataframe):
    target_column = task_detector.column_detector.detect_single_column_type(
        "query_reformulation_target", ner_dataframe
    )
    input_columns = task_detector.column_detector.get_input_columns(
        "query_reformulation_target", ner_dataframe
    )

    query_source_candidates = (
        task_detector.column_detector.get_source_column_for_query_reformulation(
            target_column, input_columns
        )
    )

    assert len(query_source_candidates) == 2

    candidates = set([x.name for x in query_source_candidates])

    assert set(["ner_target_col", "string_categorical_col"]) == candidates
