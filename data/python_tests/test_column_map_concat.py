import pytest
from dataset_utils import check_column_maps_are_equal, get_ascending_column_map


@pytest.mark.unit
def test_column_map_concat():
    ROWS = 1000
    first_half = get_ascending_column_map(ROWS)

    second_half = get_ascending_column_map(ROWS, offset=ROWS)

    concat_column_map = first_half.concat(second_half)

    assert len(concat_column_map) == ROWS * 2

    full_column_map = get_ascending_column_map(ROWS * 2)

    check_column_maps_are_equal(full_column_map, concat_column_map)

    for i in range(ROWS * 2):
        assert concat_column_map["token"][i] == i
        assert concat_column_map["token_array"][i][0] == i
        assert concat_column_map["decimal"][i] == float(i)
        assert concat_column_map["decimal_array"][i][0] == float(i)
        assert concat_column_map["string"][i] == str(i)


@pytest.mark.unit
def test_column_map_split_concat():
    ROWS = 1000
    column_map = get_ascending_column_map(ROWS * 2)

    front, back = column_map.split(ROWS)

    assert len(front) == ROWS
    assert len(back) == ROWS

    concat_column_map = front.concat(back)

    check_column_maps_are_equal(get_ascending_column_map(ROWS * 2), concat_column_map)
