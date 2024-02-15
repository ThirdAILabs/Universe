import pytest
from thirdai import data


@pytest.mark.parametrize("serialize", [True, False])
@pytest.mark.unit
def test_date_transformation(serialize):
    columns = data.ColumnMap(
        {"strings": data.columns.StringColumn(["2023-01-03", "2023-10-12"])}
    )

    transformation = data.transformations.Date("strings", "date_info")
    if serialize:
        transformation = data.transformations.deserialize(transformation.serialize())

    columns = transformation(columns)

    date_infos = columns["date_info"].data()

    feature_offset = 0

    # Day of week
    assert 0 <= date_infos[0][0] < 7
    assert 0 <= date_infos[1][0] < 7
    assert (date_infos[1][0] - date_infos[0][0]) % 7 == 2  # Should be two days apart
    feature_offset += 7  # 7 days in a week

    # Month
    assert date_infos[0][1] == (feature_offset + 0)
    assert date_infos[1][1] == (feature_offset + 9)
    feature_offset += 12  # 12 months in a year

    # Week of month
    assert date_infos[0][2] == (feature_offset + 0)
    assert date_infos[1][2] == (feature_offset + 1)
    feature_offset += 5  # up to 5 weeks in a month

    # Week of year
    assert date_infos[0][3] == (feature_offset + 0)
    assert date_infos[1][3] == (feature_offset + 40)
