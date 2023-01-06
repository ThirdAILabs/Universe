import pytest
from dataset_utils import (
    get_random_sentence_str_column,
    get_simple_dag_model,
    sparse_bolt_dataset_to_numpy,
    verify_pairgrams_distribution,
)
from thirdai import data

pytestmark = [pytest.mark.unit]

NUM_ROWS = 10000
OUTPUT_RANGE = 1000
NUM_WORDS = 5


def create_random_sentence_pairgram_dataset(prepare_for_backpropagate=False):
    sentence_column = get_random_sentence_str_column(NUM_ROWS, NUM_WORDS)

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
    columns = featurizer.featurize(columns, prepare_for_backpropagate)

    return columns, featurizer


def test_sentence_pairgrams():
    columns, _ = create_random_sentence_pairgram_dataset()
    pairgram_bolt_dataset = columns.convert_to_dataset(
        ["pairgrams"], batch_size=NUM_ROWS
    )
    pairgram_dataset = sparse_bolt_dataset_to_numpy(pairgram_bolt_dataset)
    verify_pairgrams_distribution(pairgram_dataset, OUTPUT_RANGE, NUM_WORDS)


def test_sentence_pairgrams_explanations():
    columns, featurizer = create_random_sentence_pairgram_dataset(True)
    model = get_simple_dag_model(
        input_dim=100000, hidden_layer_dim=100, hidden_layer_sparsity=1, output_dim=151
    )
    pairgram_dataset = columns.convert_to_dataset(["pairgrams"], batch_size=NUM_ROWS)

    indices, gradients = model.get_input_gradients_batch([pairgram_dataset[0]])

    assert len(indices) == NUM_ROWS
    assert len(gradients) == NUM_ROWS

    contribution_columns = columns.get_contribution_columns(
        ["pairgrams"], gradients, indices
    )

    explanations = featurizer.explain(columns, contribution_columns)

    for row in range(NUM_ROWS):
        temp = explanations.getitem("pairgrams").get_row(row)
        total_sum = 0
        for i in range(len(temp)):
            total_sum += abs(temp[i].gradient)

        assert total_sum > 99.99 and total_sum < 100.01
