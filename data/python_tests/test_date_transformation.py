import pytest
from thirdai import data


@pytest.mark.unit
def test_date_transformation():
    columns = data.ColumnMap(
        {"strings": data.columns.StringColumn(["2023-01-03", "2023-10-12"])}
    )

    transformation = data.transformations.Date("strings", "date_info")

    columns = transformation(columns)

    date_infos = columns["date_info"].data()

    print(date_infos)

    # Day of week
    assert (date_infos[1][0] - date_infos[0][0]) % 7 == 2

    # Month
    assert date_infos[0][1] == (7 + 0)
    assert date_infos[1][1] == (7 + 9)

    # Week of month
    assert date_infos[0][2] == (7 + 12 + 0)
    assert date_infos[1][2] == (7 + 12 + 1)

    # Week of year
    assert date_infos[0][3] == (7 + 12 + 5 + 0)
    assert date_infos[1][3] == (7 + 12 + 5 + 40)
