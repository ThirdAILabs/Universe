import pytest
from thirdai import data


@pytest.mark.unit
def test_dyadic_interval_augmentation():
    columns = data.ColumnMap(
        {
            "text": data.columns.TokenArrayColumn(
                [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [20, 21, 22, 23, 24, 25]],
                dim=100,
            )
        }
    )

    transform = data.transformations.DyadicInterval(
        input_column="text",
        output_interval_prefix="interval_",
        target_column="target",
        n_intervals=3,
    )

    columns = transform(columns)

    interval_1 = [[0], [1], [2], [3], [5], [6], [7], [8], [10], [20], [21], [22], [23]]
    assert columns["interval_from_end_1"].data() == interval_1

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
    assert columns["interval_from_end_2"].data() == interval_2

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
    assert columns["interval_from_end_4"].data() == interval_4

    target = [1, 2, 3, 4, 6, 7, 8, 9, 11, 21, 22, 23, 24]
    assert columns["target"].data() == target


@pytest.mark.unit
def test_dyadic_interval_inference():
    columns = data.ColumnMap(
        {
            "text": data.columns.TokenArrayColumn(
                [[0, 1, 2, 3, 4], [5, 6, 7], [8]],
                dim=100,
            ),
            "prompt": data.columns.TokenArrayColumn(
                [[10, 11], [12], [13]],
                dim=100,
            ),
        }
    )

    transform = data.transformations.DyadicInterval(
        input_column="text",
        prompt_column="prompt",
        output_interval_prefix="interval_",
        target_column="target",
        n_intervals=3,
    )

    columns = transform.inference_featurization(columns)

    prompt = [[10, 11], [12], [13]]
    assert columns["prompt"].data() == prompt

    interval_1 = [[4], [7], [8]]
    assert columns["interval_from_end_1"].data() == interval_1

    interval_2 = [[3, 4], [6, 7], [8]]
    assert columns["interval_from_end_2"].data() == interval_2

    interval_4 = [[1, 2, 3, 4], [5, 6, 7], [8]]
    assert columns["interval_from_end_4"].data() == interval_4


@pytest.mark.unit
def test_dyadic_interval_augmentation_bidirectional():
    columns = data.ColumnMap(
        {
            "text": data.columns.TokenArrayColumn(
                [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [20, 21, 22, 23, 24, 25]],
                dim=100,
            ),
            "prompt": data.columns.TokenArrayColumn(
                [[19, 20], [51, 52]],
                dim=100,
            ),
        }
    )

    transform = data.transformations.DyadicInterval(
        input_column="text",
        prompt_column="prompt",
        output_interval_prefix="interval_",
        target_column="target",
        n_intervals=3,
        is_bidirectional=True,
    )

    columns = transform(columns)

    prompt = [
        [19, 20],
        [19, 20],
        [19, 20],
        [19, 20],
        [19, 20],
        [19, 20],
        [19, 20],
        [19, 20],
        [19, 20],
        [51, 52],
        [51, 52],
        [51, 52],
        [51, 52],
    ]
    assert columns["prompt"].data() == prompt

    interval_1 = [[0], [0], [0], [0], [5], [5], [5], [5], [10], [20], [20], [20], [20]]
    assert columns["interval_from_start_1"].data() == interval_1

    interval_2 = [
        [0],
        [0, 1],
        [0, 1],
        [0, 1],
        [5],
        [5, 6],
        [5, 6],
        [5, 6],
        [10],
        [20],
        [20, 21],
        [20, 21],
        [20, 21],
    ]
    assert columns["interval_from_start_2"].data() == interval_2

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
    assert columns["interval_from_start_4"].data() == interval_4

    target = [1, 2, 3, 4, 6, 7, 8, 9, 11, 21, 22, 23, 24]
    assert columns["target"].data() == target

@pytest.mark.unit
def test_dyadic_interval_with_context():
    columns = data.ColumnMap(
        {
            "context": data.columns.TokenArrayColumn(
                [[100, 101, 102], [104]],
                dim=100,
            )
            "text": data.columns.TokenArrayColumn(
                [[0, 1, 2, 3, 4], [5, 6, 7], [8]],
                dim=100,
            )
        }
    )

    transform = data.transformations.DyadicInterval(
        input_column="text",
        context_column="context",
        output_interval_prefix="interval_",
        target_column="target",
        n_intervals=3,
        is_bidirectional=True,
    )

    columns = transform.inference_featurization(columns)

    interval_1 = [[100], [104], [8]]
    assert columns["interval_from_start_1"].data() == interval_1

    interval_2 = [[100, 101], [104, 5], [8]]
    assert columns["interval_from_start_2"].data() == interval_2

    interval_4 = [[100, 101, 102, 0], [104, 5, 6], [8]]
    assert columns["interval_from_start_4"].data() == interval_4
    
    target = [1, 4, 7, 8]
    assert columns["target"].data() == target
    
@pytest.mark.unit
def test_dyadic_interval_inference_bidirectional():
    columns = data.ColumnMap(
        {
            "text": data.columns.TokenArrayColumn(
                [[0, 1, 2, 3, 4], [5, 6, 7], [8]],
                dim=100,
            )
        }
    )

    transform = data.transformations.DyadicInterval(
        input_column="text",
        output_interval_prefix="interval_",
        target_column="target",
        n_intervals=3,
        is_bidirectional=True,
    )

    columns = transform.inference_featurization(columns)

    interval_1 = [[0], [5], [8]]
    assert columns["interval_from_start_1"].data() == interval_1

    interval_2 = [[0, 1], [5, 6], [8]]
    assert columns["interval_from_start_2"].data() == interval_2

    interval_4 = [[0, 1, 2, 3], [5, 6, 7], [8]]
    assert columns["interval_from_start_4"].data() == interval_4
