import pytest
from test_sentence_unigram import get_sentence_str_column
from test_string_hash_transformation import get_str_col
from thirdai import new_dataset as dataset

pytestmark = [pytest.mark.unit]

NUM_ROWS = 10000
OUTPUT_RANGE = 100000


def cross_column_pairgram_dataset():
    num_cols = 3
    string_columns = [get_str_col(NUM_ROWS) for _ in range(num_cols)]

    columns = dataset.ColumnMap(
        {f"column{i}": string_columns[i] for i in range(num_cols)}
    )

    featurizer = dataset.FeaturizationPipeline(
        transformations=[
            dataset.transformations.StringHash(
                input_column=column_name,
                output_column=f"{column_name}_hashes",
                output_range=OUTPUT_RANGE,
            )
            for column_name in ["column1", "column2", "column3"]
        ]
        + [
            dataset.transformations.CrossColumnPairgram(
                input_columns=[
                    col_name for col_name in ["column1", "column2", "column3"]
                ],
                output_column="pairgrams",
                output_range=OUTPUT_RANGE,
            )
        ]
    )

    columns = featurizer.featurize(columns)

    return columns.convert_to_dataset(["pairgrams"], batch_size=NUM_ROWS)


def sentence_pairgram_dataset():
    sentence_column = get_sentence_str_column(NUM_ROWS)

    columns = dataset.ColumnMap({"sentence": sentence_column})

    featurizer = dataset.FeaturizationPipeline(
        transformations=[
            dataset.transformations.SentenceUnigram(
                input_column="sentence",
                output_column="unigrams",
                output_range=OUTPUT_RANGE,
                deduplicate=False,
            ),
            dataset.transformations.TokenPairgram(
                input_column="unigrams",
                output_column="pairgrams",
                output_range=OUTPUT_RANGE,
                deduplicate=True,
            ),
        ]
    )
    columns = featurizer.featurize(columns)
    return columns.convert_to_dataset(["pairgrams"], batch_size=NUM_ROWS)


def verify_pairgrams(pairgram_dataset):
    pass


def test_cross_column_pairgrams():
    pairgram_dataset = cross_column_pairgram_dataset()
    verify_pairgrams(pairgram_dataset)


def test_sentence_pairgrams():
    pairgram_dataset = sentence_pairgram_dataset()
    verify_pairgrams(pairgram_dataset)
