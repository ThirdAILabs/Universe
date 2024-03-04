import pytest
from thirdai import data


@pytest.mark.parametrize("serialize", [True, False])
@pytest.mark.unit
def test_splade_featurization(serialize):
    columns = data.ColumnMap(
        {
            "text": data.columns.TokenArrayColumn(
                [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [20, 21, 22, 23, 24, 25]],
                dim=100,
            )
        }
    )

    transform = data.transformations.SpladeFeaturizer(
        context_length=16,
        fill_empty_contexts=False,
        source_column="text",
        partition_length=4,
        output_interval_prefix="output",
    )
    if serialize:
        transform = data.transformations.deserialize(transform.serialize())

    columns = transform(columns)
    interval_1 = [[0, 1, 2, 3], [20, 21, 22, 23]]
    assert columns["output_1"].data() == interval_1
    interval_2 = [[4, 5, 6, 7], [24, 25]]
    assert columns["output_2"].data() == interval_2
    interval_3 = [[8, 9, 10, 11], []]
    assert columns["output_3"].data() == interval_3
    interval_4 = [[], []]
    assert columns["output_4"].data() == interval_4


@pytest.mark.parametrize("serialize", [True, False])
@pytest.mark.unit
def test_splade_featurization_with_fill_values(serialize):
    columns = data.ColumnMap(
        {
            "text": data.columns.TokenArrayColumn(
                [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [20, 21, 22, 23, 24, 25]],
                dim=100,
            )
        }
    )

    transform = data.transformations.SpladeFeaturizer(
        context_length=32,
        fill_empty_contexts=True,
        source_column="text",
        partition_length=4,
        output_interval_prefix="output",
    )
    if serialize:
        transform = data.transformations.deserialize(transform.serialize())

    columns = transform(columns)
    interval_1 = [[0, 1, 2, 3], [20, 21, 22, 23]]
    assert columns["output_1"].data() == interval_1
    interval_2 = [[4, 5, 6, 7], [24, 25]]
    assert columns["output_2"].data() == interval_2
    interval_3 = [[8, 9, 10, 11], [20, 21, 22, 23]]
    assert columns["output_3"].data() == interval_3
    interval_4 = [[], [24, 25]]
    assert columns["output_4"].data() == interval_4
    interval_1 = [[0, 1, 2, 3], [20, 21, 22, 23]]
    assert columns["output_1"].data() == interval_1
    interval_2 = [[4, 5, 6, 7], [24, 25]]
    assert columns["output_2"].data() == interval_2
    interval_3 = [[8, 9, 10, 11], [20, 21, 22, 23]]
    assert columns["output_3"].data() == interval_3
    interval_4 = [[], [24, 25]]
    assert columns["output_4"].data() == interval_4
