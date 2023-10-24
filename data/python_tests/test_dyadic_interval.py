import pytest
from thirdai import data


@pytest.mark.parametrize("serialize", [True, False])
@pytest.mark.unit
def test_dyadic_interval_augmentation(serialize):
    columns = data.ColumnMap(
        {
            "text": data.columns.TokenArrayColumn(
                [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [20, 21, 22, 23, 24, 25]],
                dim=100,
            )
        }
    )

    transform = data.transformations.DyadicInterval(
        "text", "interval_", "target", n_intervals=3
    )

    if serialize:
        transform = data.transformations.deserialize(transform.serialize())

    columns = transform(columns)

    interval_1 = [[0], [1], [2], [3], [5], [6], [7], [8], [10], [20], [21], [22], [23]]
    assert columns["interval_1"].data() == interval_1

    interval_2 = [
        [0],
        [0, 1],
        [1, 2],
        [2, 3],
        [5],
        [5, 6],
        [6, 7],
        [7, 8],
        [10],
        [20],
        [20, 21],
        [21, 22],
        [22, 23],
    ]
    assert columns["interval_2"].data() == interval_2

    interval_4 = [
        [0],
        [0, 1],
        [0, 1, 2],
        [0, 1, 2, 3],
        [5],
        [5, 6],
        [5, 6, 7],
        [5, 6, 7, 8],
        [10],
        [20],
        [20, 21],
        [20, 21, 22],
        [20, 21, 22, 23],
    ]
    assert columns["interval_4"].data() == interval_4

    target = [1, 2, 3, 4, 6, 7, 8, 9, 11, 21, 22, 23, 24]
    assert columns["target"].data() == target


@pytest.mark.unit
def test_dyadic_interval_inference():
    columns = data.ColumnMap(
        {
            "text": data.columns.TokenArrayColumn(
                [[0, 1, 2, 3, 4], [5, 6, 7], [8]],
                dim=100,
            )
        }
    )

    transform = data.transformations.DyadicInterval(
        "text", "interval_", "target", n_intervals=3
    )

    columns = transform.inference_featurization(columns)

    interval_1 = [[4], [7], [8]]
    assert columns["interval_1"].data() == interval_1

    interval_2 = [[3, 4], [6, 7], [8]]
    assert columns["interval_2"].data() == interval_2

    interval_4 = [[1, 2, 3, 4], [5, 6, 7], [8]]
    assert columns["interval_4"].data() == interval_4
