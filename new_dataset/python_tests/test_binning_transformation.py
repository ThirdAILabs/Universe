import numpy as np
import pytest
from dataset_utils import get_simple_dag_model, sparse_bolt_dataset_to_numpy
from thirdai import data

pytestmark = [pytest.mark.unit]

n_rows = 100


def get_binned_featurizer(column1_np, column2_np, prepare_for_backpropagate=False):
    column1 = data.columns.DenseFeatureColumn(column1_np)
    column2 = data.columns.DenseFeatureColumn(column2_np)

    columns = data.ColumnMap({"column1": column1, "column2": column2})

    featurizer = data.FeaturizationPipeline(
        transformations=[
            data.transformations.Binning(
                input_column="column1",
                output_column="column1_binned",
                inclusive_min=0,
                exclusive_max=100,
                num_bins=5,
            ),
            data.transformations.Binning(
                input_column="column2",
                output_column="out_column2",
                inclusive_min=10,
                exclusive_max=110,
                num_bins=20,
            ),
        ]
    )

    columns = featurizer.featurize(columns, prepare_for_backpropagate)

    return columns, featurizer


def test_binning_transformation():
    column1_np = np.arange(n_rows, dtype=np.float32)
    np.random.shuffle(column1_np)

    column2_np = np.arange(start=10, stop=n_rows + 10, dtype=np.float32)
    np.random.shuffle(column2_np)
    columns, _ = get_binned_featurizer(column1_np, column2_np)

    indices, values = sparse_bolt_dataset_to_numpy(
        columns.convert_to_dataset(["column1_binned", "out_column2"], batch_size=29)
    )

    # We need to reshape the arrays so we can concatenate them on the correct axis.
    column1_bins = (np.reshape(column1_np, newshape=(n_rows, 1)) / 20).astype(np.uint32)
    column2_bins = ((np.reshape(column2_np, newshape=(n_rows, 1)) - 10) / 5).astype(
        np.uint32
    )

    expected_bins = np.concatenate(
        # We shift column2_bins by 5 to account for the dimension of the sparse values from column1.
        [column1_bins, column2_bins + 5],
        axis=1,
        dtype=np.uint32,
    )

    assert np.array_equal(indices, expected_bins)

    assert np.array_equal(values, np.ones(shape=(n_rows, 2)))


def test_binning_transformation_explanations():
    column1_np = np.arange(n_rows, dtype=np.float32)
    np.random.shuffle(column1_np)

    column2_np = np.arange(start=10, stop=n_rows + 10, dtype=np.float32)
    np.random.shuffle(column2_np)

    columns, featurizer = get_binned_featurizer(column1_np, column2_np, True)

    binned_dataset = columns.convert_to_dataset(
        ["column1_binned", "out_column2"], batch_size=n_rows
    )

    model = get_simple_dag_model(
        input_dim=100000, hidden_layer_dim=100, hidden_layer_sparsity=1, output_dim=151
    )

    indices, gradients = model.get_input_gradients_batch([binned_dataset[0]])

    contribution_columns = columns.get_contribution_columns(
        ["column1_binned", "out_column2"], gradients, indices
    )

    explanations = featurizer.explain(columns, contribution_columns)

    for row in range(n_rows):
        total_sum = abs(gradients[row][0]) + abs(gradients[row][1])
        value1 = (gradients[row][0] / total_sum) * 100
        value2 = (gradients[row][1] / total_sum) * 100
        temp1 = explanations.getitem("column1_binned").get_row(row)

        temp2 = explanations.getitem("out_column2").get_row(row)

        assert abs(value1 - temp1[0].gradient) < 0.0001
        assert abs(value2 - temp2[0].gradient) < 0.0001
