import pytest
from thirdai import data


@pytest.mark.parametrize("serialize", [True, False])
@pytest.mark.unit
def test_dyadic_contrastive_augmentation(serialize):
    columns = data.ColumnMap(
        {
            "text_1": data.columns.TokenArrayColumn(
                [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [20, 21, 22, 23, 24, 25]],
                dim=100,
            ),
            "text_2": data.columns.TokenArrayColumn(
                [
                    [20, 21, 22, 23, 24, 25, 26, 27],
                    [0, 1, 2, 3, 4, 5],
                ],
                dim=100,
            ),
            "target": data.columns.StringColumn(["0", "1"]),
            "prompt": data.columns.TokenArrayColumn(
                [
                    [15, 16, 17],
                    [11, 16, 17],
                ],
                dim=100,
            ),
        }
    )

    transform = data.transformations.ToDecimals(
        input_column="target",
        output_column="target",
    )

    columns = transform(columns)

    transform = data.transformations.DyadicContrastiveFeaturizer(
        input_column_1="text_1",
        input_column_2="text_2",
        label_column="target",
        prompt_column="prompt",
        output_interval_prefix="interval_",
        n_intervals=5,
        n_classes=2,
        is_bidirectional=True,
    )
    if serialize:
        transform = data.transformations.deserialize(transform.serialize())

    columns = transform(columns)

    interval_from_end_1_1 = [[11], [25]]
    assert columns["interval_from_end_1_1"].data() == interval_from_end_1_1

    interval_from_end_2_1 = [[10, 11], [24, 25]]
    assert columns["interval_from_end_2_1"].data() == interval_from_end_2_1

    interval_from_end_4_1 = [[8, 9, 10, 11], [22, 23, 24, 25]]
    assert columns["interval_from_end_4_1"].data() == interval_from_end_4_1

    interval_from_end_8_1 = [[4, 5, 6, 7, 8, 9, 10, 11], [20, 21, 22, 23, 24, 25]]
    assert columns["interval_from_end_8_1"].data() == interval_from_end_8_1

    interval_from_end_16_1 = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [20, 21, 22, 23, 24, 25],
    ]
    assert columns["interval_from_end_16_1"].data() == interval_from_end_16_1

    interval_from_end_1_2 = [[27], [5]]
    assert columns["interval_from_end_1_2"].data() == interval_from_end_1_2

    interval_from_end_2_2 = [[26, 27], [4, 5]]
    assert columns["interval_from_end_2_2"].data() == interval_from_end_2_2

    interval_from_end_4_2 = [[24, 25, 26, 27], [2, 3, 4, 5]]
    assert columns["interval_from_end_4_2"].data() == interval_from_end_4_2

    interval_from_end_8_2 = [[20, 21, 22, 23, 24, 25, 26, 27], [0, 1, 2, 3, 4, 5]]
    assert columns["interval_from_end_8_2"].data() == interval_from_end_8_2

    interval_from_end_16_2 = [
        [20, 21, 22, 23, 24, 25, 26, 27],
        [0, 1, 2, 3, 4, 5],
    ]
    assert columns["interval_from_end_16_2"].data() == interval_from_end_16_2

    interval_from_start_1_1 = [[0], [20]]
    assert columns["interval_from_start_1_1"].data() == interval_from_start_1_1

    interval_from_start_2_1 = [[0, 1], [20, 21]]
    assert columns["interval_from_start_2_1"].data() == interval_from_start_2_1

    interval_from_start_4_1 = [[0, 1, 2, 3], [20, 21, 22, 23]]
    assert columns["interval_from_start_4_1"].data() == interval_from_start_4_1

    interval_from_start_8_1 = [[0, 1, 2, 3, 4, 5, 6, 7], [20, 21, 22, 23, 24, 25]]
    assert columns["interval_from_start_8_1"].data() == interval_from_start_8_1

    interval_from_start_16_1 = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [20, 21, 22, 23, 24, 25],
    ]
    assert columns["interval_from_start_16_1"].data() == interval_from_start_16_1

    interval_from_start_1_2 = [[20], [0]]
    assert columns["interval_from_start_1_2"].data() == interval_from_start_1_2

    interval_from_start_2_2 = [[20, 21], [0, 1]]
    assert columns["interval_from_start_2_2"].data() == interval_from_start_2_2

    interval_from_start_4_2 = [[20, 21, 22, 23], [0, 1, 2, 3]]
    assert columns["interval_from_start_4_2"].data() == interval_from_start_4_2

    interval_from_start_8_2 = [[20, 21, 22, 23, 24, 25, 26, 27], [0, 1, 2, 3, 4, 5]]
    assert columns["interval_from_start_8_2"].data() == interval_from_start_8_2

    interval_from_start_16_2 = [
        [20, 21, 22, 23, 24, 25, 26, 27],
        [0, 1, 2, 3, 4, 5],
    ]
    assert columns["interval_from_start_16_2"].data() == interval_from_start_16_2

    target = [0, 1]
    assert columns["target"].data() == target

    prompt = [
        [15, 16, 17],
        [11, 16, 17],
    ]
    assert columns["prompt"].data() == prompt
