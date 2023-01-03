import pytest
from dataset_utils import get_simple_dag_model, sparse_bolt_dataset_to_numpy
from thirdai import bolt, data

pytestmark = [pytest.mark.unit]


def get_str_col(col_length):
    return data.columns.StringColumn([f"value{i}" for i in range(col_length)])


def get_two_col_hashed_string_dataset(col_length, output_range):

    column1, column2 = get_str_col(col_length), get_str_col(col_length)

    columns = data.ColumnMap({"column1": column1, "column2": column2})

    featurizer = data.FeaturizationPipeline(
        transformations=[
            data.transformations.StringHash(
                input_column=column_name,
                output_column=f"{column_name}_hashes",
                output_range=output_range,
            )
            for column_name in ["column1", "column2"]
        ]
    )

    return featurizer, columns


def test_string_explanations():
    col_length = 10000
    output_range = 10000
    featurizer, columns = get_two_col_hashed_string_dataset(col_length, output_range)
    columns = featurizer.featurize(columns, True)
    string_dataset = columns.convert_to_dataset(
        ["column1_hashes", "column2_hashes"], batch_size=col_length
    )
    model = get_simple_dag_model(
        input_dim=100000, hidden_layer_dim=100, hidden_layer_sparsity=1, output_dim=151
    )

    indices, gradients = model.get_input_gradients_batch([string_dataset[0]])

    assert len(indices) == col_length
    assert len(gradients) == col_length

    contribution_columns = columns.get_contribution_columns(
        ["column1_hashes", "column2_hashes"], gradients, indices
    )

    explanations = featurizer.explain(columns, contribution_columns)

    column_names = ["column1", "column2"]

    for i in range(col_length):
        for column_name in column_names:
            sw = explanations.getitem(column_name).get_row(i)

            sw1 = explanations.getitem(column_name + "_hashes").get_row(i)

            for j in range(len(sw)):

                assert sw[j].gradient == sw1[j].gradient


# Tests that if we hash two columns and then turn them into a dataset, the sparse
# indices module the output range of the first column will be the same (this
# ensures that the hash function is consistent).
def test_string_hash_consistency():
    col_length = 100
    output_range = 100

    featurizer, columns = get_two_col_hashed_string_dataset(col_length, output_range)

    columns = featurizer.featurize(columns)

    string_dataset = columns.convert_to_dataset(
        ["column1_hashes", "column2_hashes"], batch_size=col_length
    )

    indices, _ = sparse_bolt_dataset_to_numpy(string_dataset)

    for i1, i2 in indices:
        assert i1 + output_range == i2


# Tests that each hash has about the number of values we expect (is within a
# factor of 2 of the expected count). This won't be flaky because the hash is
# seeded and thus entirely deterministic.
def test_string_hash_distribution():
    col_length = 10000
    output_range = 100

    featurizer, columns = get_two_col_hashed_string_dataset(col_length, output_range)

    columns = featurizer.featurize(columns)

    string_dataset = columns.convert_to_dataset(
        ["column1_hashes", "column2_hashes"], batch_size=col_length
    )

    indices, _ = sparse_bolt_dataset_to_numpy(string_dataset)

    hash_counts = [0 for _ in range(output_range)]
    for i1, _ in indices:
        hash_counts[i1] += 1

    expected_count = col_length / output_range
    for count in hash_counts:
        assert count / expected_count < 2 and count / expected_count > 0.5
