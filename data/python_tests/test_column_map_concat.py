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
