import numpy as np
import pytest
from thirdai import data

pytestmark = [pytest.mark.unit]


@pytest.mark.parametrize("serialize", [True, False])
def test_cast_string_to_token(serialize):
    string_col = data.columns.StringColumn([str(i) for i in range(10)])
    columns = data.ColumnMap({"strings": string_col})

    transformation = data.transformations.ToTokens("strings", "tokens")
    if serialize:
        transformation = data.transformations.deserialize(transformation.serialize())

    columns = transformation(columns)

    for i in range(10):
        assert columns["tokens"][i] == i

    with pytest.raises(
        ValueError,
        match=r"Invalid index 5 for TokenColumn with dimension 5",
    ):
        data.transformations.ToTokens("strings", "tokens", dim=5)(columns)


@pytest.mark.parametrize("serialize", [True, False])
def test_cast_string_to_token_array(serialize):
    string_col = data.columns.StringColumn(
        [" ".join(map(str, range(i, i + 2))) for i in range(10)]
    )
    columns = data.ColumnMap({"strings": string_col})

    transformation = data.transformations.ToTokenArrays(
        "strings", "tokens", delimiter=" "
    )
    if serialize:
        transformation = data.transformations.deserialize(transformation.serialize())

    columns = transformation(columns)
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


@pytest.mark.parametrize("serialize", [True, False])
def test_cast_string_to_decimal(serialize):
    string_col = data.columns.StringColumn([str(0.1 * i) for i in range(10)])
    columns = data.ColumnMap({"strings": string_col})

    transformation = data.transformations.ToDecimals("strings", "decimals")
    if serialize:
        transformation = data.transformations.deserialize(transformation.serialize())

    columns = transformation(columns)

    for i in range(10):
        assert np.allclose([columns["decimals"][i]], [0.1 * i])


@pytest.mark.parametrize("serialize", [True, False])
def test_cast_string_to_decimal_array(serialize):
    string_col = data.columns.StringColumn(
        [" ".join([str(0.1 * j) for j in range(i, i + 2)]) for i in range(10)]
    )
    columns = data.ColumnMap({"strings": string_col})

    transformation = data.transformations.ToDecimalArrays(
        "strings", "decimals", delimiter=" "
    )
    if serialize:
        transformation = data.transformations.deserialize(transformation.serialize())

    columns = transformation(columns)

    for i in range(10):
        assert np.allclose(columns["decimals"][i], [i * 0.1, (i + 1) * 0.1])


@pytest.mark.parametrize(
    "conversion",
    [data.transformations.ToTokens, data.transformations.ToDecimals],
)
def test_invalid_row_in_value_column(conversion):
    columns = data.ColumnMap({"strings": data.columns.StringColumn(["1", "hello"])})

    transformation = conversion("strings", "values")
    with pytest.raises(ValueError, match="Invalid row 'hello' in column 'strings'."):
        transformation(columns)


@pytest.mark.parametrize(
    "conversion",
    [data.transformations.ToTokenArrays, data.transformations.ToDecimalArrays],
)
def test_invalid_row_in_array_column(conversion):
    columns = data.ColumnMap({"strings": data.columns.StringColumn(["1,2", "1,hello"])})

    transformation = conversion("strings", "arrays", delimiter=",")

    with pytest.raises(ValueError, match="Invalid row '1,hello' in column 'strings'."):
        transformation(columns)


@pytest.mark.parametrize("serialize", [True, False])
def test_cast_string_to_timestamp(serialize):
    string_col = data.columns.StringColumn(
        ["2023-07-" + str(date) for date in range(10, 20)]
    )
    columns = data.ColumnMap({"strings": string_col})

    transformation = data.transformations.ToTimestamps("strings", "timestamps")
    if serialize:
        transformation = data.transformations.deserialize(transformation.serialize())

    columns = transformation(columns)

    SECONDS_IN_A_DAY = 24 * 3600

    for i in range(1, 10):
        assert (
            columns["timestamps"][i] - columns["timestamps"][i - 1] == SECONDS_IN_A_DAY
        )
