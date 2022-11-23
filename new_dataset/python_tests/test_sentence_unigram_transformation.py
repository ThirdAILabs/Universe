from collections import defaultdict

import pytest
from thirdai import data

pytestmark = [pytest.mark.unit]


def get_sentence_str_column(col_length):
    return data.columns.StringColumn(
        [f"value{i} value{i} value{i+1}" for i in range(col_length)]
    )


# tests that deduplication in the sentence unigram block doesn't actually change
# the resulting boltvector
def test_sentence_unigram_deduplication():
    col_length = 10000
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
