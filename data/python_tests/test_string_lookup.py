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


def test_reusing_vocab():
    ROWS = 1000

    strings = [f"str_{i}" for i in range(ROWS)]

    columns = data.ColumnMap({"strings": data.columns.StringColumn(strings)})

    transformation = data.transformations.StringLookup(
        "strings", "ids", vocab_key="ids", delimiter=","
    )

    state = data.transformations.State()

    columns = transformation(columns, state)

    original_ids = columns["ids"].data()

    columns = data.ColumnMap(
        {"strings": data.columns.StringColumn(strings[ROWS // 2 :])}
    )

    new_transformation = data.transformations.deserialize(transformation.serialize())
    columns = new_transformation(columns, state)

    new_ids = columns["ids"].data()

    assert set(x[0] for x in original_ids) == set(range(ROWS))

    # The new column is the second half of the original column so the ids should
    # match since the transformation is reused.
    assert new_ids == original_ids[ROWS // 2 :]

    # This is to check that the transformation is using the same vocab and not creating
    # a new mapping of strings to ids. The vocabulary will use consecutive integers
    # when created, so the original column maps strings to ids in [0, ROWS). Due
    # to parallelism the second half of the column won't have exactly the ids
    # [ROWS / 2, ROWS), but it still should have some ids from this range. If the
    # transformation was applied to the second half of the column with a new vocab
    # then the ids would be [0, ROWS / 2).
    assert max(x[0] for x in new_ids) > len(new_ids)
