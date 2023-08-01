import pytest
from thirdai import data

pytestmark = [pytest.mark.unit]


def test_repeating_values():
    columns = data.ColumnMap(
        {"strings": data.columns.StringColumn(["a", "b", "b,c", "a", "a,c", "c"])}
    )

    transformation = data.transformations.StringLookup(
        "strings", "ids", vocab_key="ids", delimiter=","
    )

    columns = transformation(columns)

    ids = columns["ids"].data()

    a_id = ids[0][0]
    b_id = ids[1][0]
    c_id = ids[5][0]

    assert set([a_id, b_id, c_id]) == set([0, 1, 2])

    expected = [[a_id], [b_id], [b_id, c_id], [a_id], [a_id, c_id], [c_id]]

    assert ids == expected


def test_large_column():
    ROWS = 1000

    columns = data.ColumnMap(
        {
            "strings": data.columns.StringColumn(
                [f"str_{i},str_{(i+1)%ROWS}" for i in range(ROWS)]
            )
        }
    )

    transformation = data.transformations.StringLookup(
        "strings", "ids", vocab_key="ids", delimiter=","
    )

    columns = transformation(columns)

    ids = columns["ids"].data()

    assert set(x[0] for x in ids) == set(range(ROWS))

    for i in range(ROWS):
        assert ids[i][1] == ids[(i + 1) % ROWS][0]
