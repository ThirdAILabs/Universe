import pytest
import random
import string
from dataset_utils import sparse_bolt_dataset_to_numpy
from thirdai import new_dataset as dataset

pytestmark = [pytest.mark.unit]

NUM_ROWS = 10000
OUTPUT_RANGE = 100000


def random_word(length=5):
    return "".join(random.choice(string.ascii_lowercase) for _ in range(length))


def random_sentence(num_words=5):
    return " ".join(random_word() for _ in range(num_words))


def get_str_col(col_length):
    return dataset.columns.StringColumn([random_word() for _ in range(col_length)])


def get_sentence_str_column(col_length):
    return dataset.columns.StringColumn([random_sentence() for _ in range(col_length)])


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
            for column_name in ["column0", "column1", "column2"]
        ]
        + [
            dataset.transformations.CrossColumnPairgram(
                input_columns=[
                    f"{col_name}_hashes"
                    for col_name in ["column0", "column1", "column2"]
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
            ),
        ]
    )
    columns = featurizer.featurize(columns)
    return columns.convert_to_dataset(["pairgrams"], batch_size=NUM_ROWS)


def verify_pairgrams(pairgram_dataset):
    indices, _ = pairgram_dataset
    hash_counts = [0 for _ in range(OUTPUT_RANGE)]
    for row in indices:
        for index in row:
            hash_counts[index] += 1

    expected_count = NUM_ROWS / OUTPUT_RANGE
    for count in hash_counts:
        assert count / expected_count < 2 or count / expected_count > 0.5


def test_cross_column_pairgrams():
    pairgram_dataset = cross_column_pairgram_dataset()
    verify_pairgrams(pairgram_dataset)


def test_sentence_pairgrams():
    pairgram_dataset = sparse_bolt_dataset_to_numpy(sentence_pairgram_dataset())
    verify_pairgrams(pairgram_dataset)
