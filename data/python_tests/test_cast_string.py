import numpy as np
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
        data.transformations.ToTokens("strings", "tokens", dim=5)(columns)


def test_cast_string_to_token_array():
    string_col = data.columns.StringColumn(
        [" ".join(map(str, range(i, i + 2))) for i in range(10)]
    )
    columns = data.ColumnMap({"strings": string_col})
    columns = data.transformations.ToTokenArrays("strings", "tokens", delimiter=" ")(
        columns
    )
    for i in range(10):
        assert columns["tokens"][i][0] == i
        assert columns["tokens"][i][1] == i + 1

    with pytest.raises(
        ValueError,
        match=r"Invalid index 5 for TokenArrayColumn with dimension 5",
    ):
        data.transformations.ToTokenArrays("strings", "tokens", delimiter=" ", dim=5)(
            columns
        )


def test_cast_string_to_decimal():
    string_col = data.columns.StringColumn([str(0.1 * i) for i in range(10)])
    columns = data.ColumnMap({"strings": string_col})
    columns = data.transformations.ToDecimals("strings", "decimals")(columns)
    for i in range(10):
        assert np.allclose([columns["decimals"][i]], [0.1 * i])


def test_cast_string_to_decimal_array():
    string_col = data.columns.StringColumn(
        [" ".join([str(0.1 * j) for j in range(i, i + 2)]) for i in range(10)]
    )
    columns = data.ColumnMap({"strings": string_col})
    columns = data.transformations.ToDecimalArrays(
        "strings", "decimals", delimiter=" "
    )(columns)
    for i in range(10):
        assert np.allclose(columns["decimals"][i], [i * 0.1, (i + 1) * 0.1])
