import numpy as np
import pytest
from dataset_utils import sparse_bolt_dataset_to_numpy
from thirdai import new_dataset as dataset

pytestmark = [pytest.mark.unit]


def test_binning_transformation():
    n_rows = 100

    column1_np = np.arange(n_rows, dtype=np.float32)
    np.random.shuffle(column1_np)

    column2_np = np.arange(start=10, stop=n_rows + 10, dtype=np.float32)
    np.random.shuffle(column2_np)

    column1 = dataset.columns.NumpyDenseValueColumn(column1_np)
    column2 = dataset.columns.NumpyDenseValueColumn(column2_np)

    columns = dataset.ColumnMap({"column1": column1, "column2": column2})

    featurizer = dataset.FeaturizationPipeline(
        transformations=[
            dataset.transformations.Binning(
                input_column="column1",
                output_column="column1_binned",
                inclusive_min=0,
                exclusive_max=100,
                num_bins=5,
            ),
            dataset.transformations.Binning(
                input_column="column2",
                output_column="column2",
                inclusive_min=10,
                exclusive_max=110,
                num_bins=20,
            ),
        ]
    )

    columns = featurizer.featurize(columns)

    indices, values = sparse_bolt_dataset_to_numpy(
        columns.convert_to_dataset(["column1_binned", "column2"], batch_size=29)
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
