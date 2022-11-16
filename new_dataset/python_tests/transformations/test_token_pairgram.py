import pytest
from dataset_utils import (
    get_sentence_str_column,
    sparse_bolt_dataset_to_numpy,
    verify_pairgrams,
)
from thirdai import data

NUM_ROWS = 10000
OUTPUT_RANGE = 1000
NUM_WORDS = 5


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


def test_sentence_pairgrams():
    pairgram_dataset = sparse_bolt_dataset_to_numpy(sentence_pairgram_dataset())
    verify_pairgrams(pairgram_dataset)
