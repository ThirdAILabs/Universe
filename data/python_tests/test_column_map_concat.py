import pytest
from dataset_utils import get_ascending_column_map


@pytest.mark.unit
def test_column_map_concat():
    ROWS = 1000
    first_half = get_ascending_column_map(ROWS)

    second_half = get_ascending_column_map(ROWS, offset=ROWS)

    concat_column_map = first_half.concat(second_half)

    full_column_map = get_ascending_column_map(ROWS * 2)

    assert full_column_map["token"].data() == concat_column_map["token"].data()

    for i in range(ROWS * 2):
        assert concat_column_map["token"][i] == i
        assert concat_column_map["token_array"][i][0] == i
        assert concat_column_map["decimal"][i] == float(i)
        assert concat_column_map["decimal_array"][i][0] == float(i)
