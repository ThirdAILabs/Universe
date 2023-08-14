import pytest
from dataset_utils import get_ascending_column_map


@pytest.mark.unit
def test_column_map_shuffle():
    ROWS = 1000
    columns = get_ascending_column_map(ROWS)

    columns.shuffle()

    # Check the that the columns have the same correspondence.
    for i in range(ROWS):
        assert float(columns["token"][i]) == columns["decimal"][i]
        assert str(columns["token"][i]) == columns["string"][i]

        assert columns["token_array"][i][0] == columns["token"][i]
        assert columns["decimal_array"][i][0] == columns["decimal"][i]

        for j in range(4):
            assert float(columns["token_array"][i][j]) == columns["decimal_array"][i][j]

    # Check that all rows are present.
    assert set(columns["token"].data()) == set(range(ROWS))

    # Check that the rows are actually shuffled.
    permutation = columns["token"].data()
    assert len(permutation) == ROWS

    not_shuffled = 0
    for i, p in enumerate(permutation):
        if i == p:
            not_shuffled += 1

    # << 1% of rows should not be shuffled, using 5% to ensure test is not flaky.
    assert not_shuffled / ROWS < 0.05
