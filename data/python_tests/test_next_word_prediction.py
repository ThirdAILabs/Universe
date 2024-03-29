import pytest
from thirdai import data


@pytest.mark.parametrize("serialize", [True, False])
@pytest.mark.unit
def test_next_word_prediction(serialize):
    columns = data.ColumnMap(
        {
            "text": data.columns.TokenArrayColumn(
                [[0, 1, 2, 3, 4, 5], [20, 21, 22, 23, 24, 25]],
                dim=100,
            )
        }
    )

    transform = data.transformations.NextWordPrediction(
        input_column="text",
        context_column="context",
        target_column="target",
    )
    if serialize:
        transform = data.transformations.deserialize(transform.serialize())

    columns = transform(columns)

    assert columns["context"].data() == [
        [0],
        [0, 1],
        [0, 1, 2],
        [0, 1, 2, 3],
        [0, 1, 2, 3, 4],
        [20],
        [20, 21],
        [20, 21, 22],
        [20, 21, 22, 23],
        [20, 21, 22, 23, 24],
    ]
    assert columns["target"].data() == [1, 2, 3, 4, 5, 21, 22, 23, 24, 25]
