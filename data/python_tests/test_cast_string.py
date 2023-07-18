import pytest
from thirdai import data

pytestmark = [pytest.mark.unit]


def test_cast_string_to_token():
    string_col = data.columns.StringColumn([str(i) for i in range(10)])
    columns = data.ColumnMap({"strings": string_col})
    columns = data.transformations.ToTokens("strings", "tokens")(columns)
    for i in range(10):
        assert columns["tokens"][i] == i

    with pytest.raises(
        ValueError,
        match=r"Invalid index 5 for TokenColumn with dimension 5",
    ):
        data.transformation.ToTokens("strings", "tokens", dim=5)(columns)


def test_cast_string_to_token_array():
    string_col = data.columns.StringColumn(
        [" ".join(map(str, range(i, i + 2))) for i in range(10)]
    )
    columns = data.ColumnMap({"strings": string_col})
    columns = data.transformations.ToTokenArrays("strings", "tokens")(columns)
    for i in range(10):
        assert columns["tokens"][i][0] == i
        assert columns["tokens"][i][1] == i + 1

    with pytest.raises(
        ValueError,
        match=r"Invalid index 5 for TokenArrayColumn with dimension 5",
    ):
        data.transformation.ToTokenArrays("strings", "tokens", dim=5)(columns)


def test_cast_string_to_decimal():
    string_col = data.columns.StringColumn([str(0.1 * i) for i in range(10)])
    columns = data.ColumnMap({"strings": string_col})
    columns = data.transformations.ToDecimals("strings", "decimals")(columns)
    for i in range(10):
        assert columns["decimals"][i] == 0.1 * i


def test_cast_string_to_decimal_array():
    string_col = data.columns.StringColumn(
        [" ".join([str(0.1 * j) for j in range(i, i + 2)]) for i in range(10)]
    )
    columns = data.ColumnMap({"strings": string_col})
    columns = data.transformations.ToDecimalArrays("strings", "decimals")(columns)
    for i in range(10):
        assert columns["tokens"][i][0] == i * 0.1
        assert columns["tokens"][i][1] == (i + 1) * 0.1

    with pytest.raises(
        ValueError,
        match=r"Expected consistent dimension in DecimalArrayColumn.",
    ):
        string_col = data.columns.StringColumn(
            [" ".join([str(0.1 * j) for j in range(0, i)]) for i in range(10)]
        )
        columns = data.ColumnMap({"strings": string_col})
        columns = data.transformations.ToDecimalArrays("strings", "decimals")(columns)


def test_cast_string_to_timestamp():
    string_col = data.columns.StringColumn(
        ["2023-07-" + str(date) for date in range(10, 20)]
    )
    columns = data.ColumnMap({"strings": string_col})
    columns = data.transformations.ToDecimalArrays("strings", "timestamps")(columns)

    SECONDS_IN_A_DAY = 24 * 3600

    for i in range(1, 10):
        assert (
            columns["timestamps"][i] - columns["timestamps"][i - 1] == SECONDS_IN_A_DAY
        )
