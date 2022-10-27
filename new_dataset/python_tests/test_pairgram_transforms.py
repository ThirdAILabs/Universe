import random
import string

import pytest
from dataset_utils import sparse_bolt_dataset_to_numpy
from thirdai import data

pytestmark = [pytest.mark.unit]

NUM_ROWS = 10000
OUTPUT_RANGE = 1000
NUM_WORDS = 5


def random_word(length=4):
    return "".join(random.choice(string.ascii_lowercase) for _ in range(length))


def random_sentence(num_words=NUM_WORDS):
    return " ".join(random_word() for _ in range(num_words))


def get_str_col(col_length):
    return data.columns.StringColumn([random_word() for _ in range(col_length)])


def get_sentence_str_column(col_length):
    return data.columns.StringColumn([random_sentence() for _ in range(col_length)])


def cross_column_pairgram_dataset():
    num_cols = NUM_WORDS
    string_columns = [get_str_col(NUM_ROWS) for _ in range(num_cols)]

    columns = data.ColumnMap({f"column{i}": string_columns[i] for i in range(num_cols)})

    column_name_list = [f"column{i}" for i in range(num_cols)]

    featurizer = data.FeaturizationPipeline(
        transformations=[
            data.transformations.StringHash(
                input_column=column_name,
                output_column=f"{column_name}_hashes",
            )
            for column_name in column_name_list
        ]
        + [
            data.transformations.CrossColumnPairgram(
                input_columns=[f"{col_name}_hashes" for col_name in column_name_list],
                output_column="pairgrams",
                output_range=OUTPUT_RANGE,
            )
        ]
    )

    columns = featurizer.featurize(columns)

    return columns.convert_to_dataset(["pairgrams"], batch_size=NUM_ROWS)


def sentence_pairgram_dataset():
    sentence_column = get_sentence_str_column(NUM_ROWS)

    columns = data.ColumnMap({"sentence": sentence_column})

    featurizer = data.FeaturizationPipeline(
        transformations=[
            data.transformations.SentenceUnigram(
                input_column="sentence",
                output_column="unigrams",
                deduplicate=False,
            ),
            data.transformations.TokenPairgram(
                input_column="unigrams",
                output_column="pairgrams",
                output_range=OUTPUT_RANGE,
            ),
        ]
    )
    columns = featurizer.featurize(columns)
    return columns.convert_to_dataset(["pairgrams"], batch_size=NUM_ROWS)


def verify_pairgrams(pairgram_dataset):
    indices, values = pairgram_dataset
    hash_counts = [0 for _ in range(OUTPUT_RANGE)]
    for row_indices, row_values in zip(indices, values):
        for index, value in zip(row_indices, row_values):
            hash_counts[index] += value

    # unordered pairgrams have (N * (N + 1)) / 2 values
    pairgrams_per_row = (NUM_WORDS * (NUM_WORDS + 1)) / 2
    expected_count = (NUM_ROWS / OUTPUT_RANGE) * pairgrams_per_row
    for count in hash_counts:
        assert count / expected_count < 2 and count / expected_count > 0.5


def test_cross_column_pairgrams():
    pairgram_dataset = sparse_bolt_dataset_to_numpy(cross_column_pairgram_dataset())
    verify_pairgrams(pairgram_dataset)


def test_sentence_pairgrams():
    pairgram_dataset = sparse_bolt_dataset_to_numpy(sentence_pairgram_dataset())
    verify_pairgrams(pairgram_dataset)
