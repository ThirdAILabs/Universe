from collections import defaultdict

import pytest
from thirdai import data

from bolt.python_tests.utils import get_simple_dag_model

pytestmark = [pytest.mark.unit]


def get_sentence_str_column(col_length):
    return data.columns.StringColumn(
        [f"value{i} value{i} value{i+1}" for i in range(col_length)]
    )


def get_featurizer_and_columns(col_length):
    column = get_sentence_str_column(col_length)

    columns = data.ColumnMap({"sentence": column})

    output_range = 100
    featurizer = data.FeaturizationPipeline(
        transformations=[
            data.transformations.SentenceUnigram(
                input_column="sentence",
                output_column="deduplicated",
                output_range=output_range,
                deduplicate=True,
            ),
            data.transformations.SentenceUnigram(
                input_column="sentence",
                output_column="not_deduplicated",
                output_range=output_range,
                deduplicate=False,
            ),
        ]
    )

    return columns, featurizer


# tests that deduplication in the sentence unigram block doesn't actually change
# the resulting boltvector
def test_sentence_unigram_deduplication():
    col_length = 10000
    columns, featurizer = get_featurizer_and_columns(col_length)
    columns = featurizer.featurize(columns)
    deduped_dataset = columns.convert_to_dataset(
        ["deduplicated"], batch_size=col_length
    )
    not_deduped_dataset = columns.convert_to_dataset(
        ["not_deduplicated"], batch_size=col_length
    )

    for row_idx in range(col_length):
        indices, values = not_deduped_dataset[0][row_idx].to_numpy()
        expected_values = defaultdict(int)
        for index, value in zip(indices, values):
            expected_values[index] += value

        actual_indices, actual_values = deduped_dataset[0][row_idx].to_numpy()
        for actual_index, actual_value in zip(actual_indices, actual_values):
            assert expected_values[actual_index] == actual_value


def test_sentence_unigram_explanations():
    col_length = 10000
    columns, featurizer = get_featurizer_and_columns(col_length)
    columns = featurizer.featurize(columns, True)
    deduped_dataset = columns.convert_to_dataset(
        ["not_deduplicated"], batch_size=col_length
    )
    model = get_simple_dag_model(
        input_dim=100000, hidden_layer_dim=100, hidden_layer_sparsity=1, output_dim=151
    )

    org_indices, org_gradients = model.get_input_gradients_batch([deduped_dataset[0]])

    contribution_columns = columns.get_contribution_columns(
        ["not_deduplicated"], org_gradients, org_indices
    )

    explanations = featurizer.explain(columns, contribution_columns)

    for i in range(col_length):
        sentence_explain = explanations.getitem("sentence").get_row(i)
        not_deduplicated_explain = explanations.getitem("not_deduplicated").get_row(i)
        for j in range(len(sentence_explain)):
            assert sentence_explain[j].gradient == not_deduplicated_explain[j].gradient
