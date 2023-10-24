import numpy as np
import pytest
from thirdai import data

pytestmark = [pytest.mark.unit]


@pytest.mark.parametrize("serialize", [True, False])
def test_binning_transformation(serialize):
    n_rows = 100

    column1_np = np.arange(n_rows, dtype=np.float32)
    np.random.shuffle(column1_np)

    column2_np = np.arange(start=10, stop=n_rows + 10, dtype=np.float32)
    np.random.shuffle(column2_np)

    column1 = data.columns.DecimalColumn(column1_np)
    column2 = data.columns.DecimalColumn(column2_np)

    columns = data.ColumnMap({"column1": column1, "column2": column2})

    featurizer = data.transformations.TransformationList(
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
                output_column="column2_binned",
                inclusive_min=10,
                exclusive_max=110,
                num_bins=20,
            ),
        ]
    )
    if serialize:
        featurizer = data.transformations.deserialize(featurizer.serialize())

    columns = featurizer(columns)

    column1_bins = np.array(columns["column1_binned"].data())
    column2_bins = np.array(columns["column2_binned"].data())

    # We need to reshape the arrays so we can concatenate them on the correct axis.
    expected_column1_bins = (column1_np / 20).astype(np.uint32)
    expected_column2_bins = ((column2_np - 10) / 5).astype(np.uint32)

    assert np.array_equal(column1_bins, expected_column1_bins)

    assert np.array_equal(column2_bins, expected_column2_bins)
