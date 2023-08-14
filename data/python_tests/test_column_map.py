import re

import pytest
from dataset_utils import check_column_maps_are_equal, get_ascending_column_map
from thirdai import data

pytestmark = [pytest.mark.unit]


def test_missing_column():
    columns = data.ColumnMap({"a": data.columns.TokenColumn([])})

    assert len(columns["a"]) == 0

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Unable to find column with name 'b'. ColumnMap contains columns ['a']."
        ),
    ):
        columns["b"]


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


def test_column_map_concat_undoes_split():
    ROWS = 1000
    column_map = get_ascending_column_map(ROWS * 2)

    front, back = column_map.split(ROWS)

    assert len(front) == ROWS
    assert len(back) == ROWS

    concat_column_map = front.concat(back)

    check_column_maps_are_equal(get_ascending_column_map(ROWS * 2), concat_column_map)


def test_column_map_split_undoes_concat():
    ROWS = 1000
    front = get_ascending_column_map(ROWS)
    back = get_ascending_column_map(ROWS, offset=ROWS)

    concat_column_map = front.concat(back)

    assert len(concat_column_map) == ROWS * 2

    front, back = concat_column_map.split(ROWS)

    check_column_maps_are_equal(get_ascending_column_map(ROWS), front)
    check_column_maps_are_equal(get_ascending_column_map(ROWS, offset=ROWS), back)


def empty_column():
    return data.columns.StringColumn([])


def test_column_map_concat_with_itself():
    columns = data.ColumnMap({"col": empty_column()})

    with pytest.raises(ValueError, match="Cannot concatenate a ColumnMap with itself."):
        columns.concat(columns)


def test_column_map_concat_column_with_itself():
    column = empty_column()
    columns_a = data.ColumnMap({"col": column})
    columns_b = data.ColumnMap({"col": column})

    with pytest.raises(ValueError, match="Cannot concatenate a column with itself."):
        columns_a.concat(columns_b)


def test_column_map_concat_with_mismatching_columns():
    columns_a = data.ColumnMap({"a": empty_column(), "b": empty_column()})
    columns_b = data.ColumnMap(
        {"a": empty_column(), "b": empty_column(), "c": empty_column()}
    )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Cannot call concat on ColumnMaps with different columns. One ColumnMap has columns ['b', 'a'] and the other has columns ['c', 'b', 'a']."
        ),
    ):
        columns_a.concat(columns_b)

    columns_a = data.ColumnMap({"a": empty_column(), "b": empty_column()})
    columns_b = data.ColumnMap({"a": empty_column(), "c": empty_column()})

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Cannot call concat on ColumnMaps with different columns. One ColumnMap has columns ['b', 'a'] and the other has columns ['c', 'a']."
        ),
    ):
        columns_a.concat(columns_b)
