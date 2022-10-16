import numpy as np
import pytest
from thirdai import dataset

pytestmark = [pytest.mark.unit]


def test_simple_dense_columns():
    n_rows = 20

    column_1d = np.arange(n_rows)
    column_2d = np.random.rand(shape=(n_rows, 10))

    column1 = dataset.columns.NumpyFloatValueColumn(array=column_1d)
    column2 = dataset.columns.NumpyFloatArrayColumn(array=column_2d)

    featurizer = dataset.FeaturizationPipeline(transformations=[], output_columns=["column1, column2"])

    columns = dataset.ColumnMap({"column1": column1, "column2": column2})
    vectors = [v.numpy() for v in featurizer.featurize(columns)]

    assert np.array_equal(np.array(vectors), np.concatenate([column_1d, column_2d], axis=1))
    