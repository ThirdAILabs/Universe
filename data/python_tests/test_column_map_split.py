import pytest
from dataset_utils import check_column_maps_are_equal, get_ascending_column_map


@pytest.mark.unit
def test_column_map_split():
    ROWS = 1000

    column_map = get_ascending_column_map(ROWS * 2)

    front, back = column_map.split(ROWS)

    assert len(front) == ROWS
    assert len(back) == ROWS

    check_column_maps_are_equal(get_ascending_column_map(ROWS), front)
    check_column_maps_are_equal(get_ascending_column_map(ROWS, offset=ROWS), back)

    for i in range(ROWS):
        assert front["token"][i] == i
        assert back["token"][i] == (i + ROWS)

        assert front["token_array"][i][0] == i
        assert back["token_array"][i][0] == (i + ROWS)

        assert front["decimal"][i] == float(i)
        assert back["decimal"][i] == float(i + ROWS)

        assert front["decimal_array"][i][0] == float(i)
        assert back["decimal_array"][i][0] == float(i + ROWS)

        assert front["string"][i] == str(i)
        assert back["string"][i] == str(i + ROWS)


@pytest.mark.unit
def test_column_map_concat_split():
    ROWS = 1000
    front = get_ascending_column_map(ROWS)
    back = get_ascending_column_map(ROWS, offset=ROWS)

    concat_column_map = front.concat(back)

    assert len(concat_column_map) == ROWS * 2

    front, back = concat_column_map.split(ROWS)

    check_column_maps_are_equal(get_ascending_column_map(ROWS), front)
    check_column_maps_are_equal(get_ascending_column_map(ROWS, offset=ROWS), back)
