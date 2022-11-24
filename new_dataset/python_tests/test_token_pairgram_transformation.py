import pytest
from dataset_utils import (
    get_random_sentence_str_column,
    sparse_bolt_dataset_to_numpy,
    verify_pairgrams_distribution,
)
from thirdai import data

pytestmark = [pytest.mark.unit]

NUM_ROWS = 10000
OUTPUT_RANGE = 1000
NUM_WORDS = 5


def create_random_sentence_pairgram_dataset():
    sentence_column = get_random_sentence_str_column(NUM_ROWS, NUM_WORDS)

    columns = data.ColumnMap({"sentence": sentence_column})

    featurizer = data.FeaturizationPipeline(
        transformations=[
            data.transformations.SentenceUnigram(
                input_column="sentence", output_column="unigrams", deduplicate=False
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
    pairgram_dataset = sparse_bolt_dataset_to_numpy(
        create_random_sentence_pairgram_dataset()
    )
    verify_pairgrams_distribution(pairgram_dataset, OUTPUT_RANGE, NUM_WORDS)
