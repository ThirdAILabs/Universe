import random
from collections import defaultdict

import pytest
from thirdai import data

pytestmark = [pytest.mark.unit]


@pytest.mark.parametrize("include_current_row", [True, False])
def test_categorical_temporal(include_current_row):
    N_USERS = 20
    OCCURENCES_PER_USER = 10
    users = [f"user_{i}" for i in range(N_USERS)] * OCCURENCES_PER_USER
    random.shuffle(users)
    timestamps = list(range(len(users)))
    items = []
    user_occurences = defaultdict(int)
    for user in users:
        item_id = user_occurences[user]
        items.append([item_id * 2, item_id * 2 + 1])
        user_occurences[user] += 1

    columns = data.ColumnMap(
        {
            "users": data.columns.StringColumn(users),
            "items": data.columns.TokenArrayColumn(items),
            "timestamps": data.columns.TimestampColumn(timestamps),
        }
    )

    transformation = data.transformations.CategoricalTemporal(
        user_column="users",
        item_column="items",
        timestamp_column="timestamps",
        output_column="history",
        track_last_n=4,
        include_current_row=include_current_row,
    )

    columns = transformation(columns)

    user_histories = defaultdict(list)

    for i in range(len(columns)):
        user_histories[columns["users"][i]].append(list(columns["history"][i]))

    assert len(user_histories) == N_USERS

    for user, histories in user_histories.items():
        assert len(histories) == OCCURENCES_PER_USER

        for i, h in enumerate(histories):
            if include_current_row:
                expected = list(reversed(range(max(0, (i * 2) - 2), (i * 2) + 2)))
                last = expected[0]
                expected[0] = expected[1]
                expected[1] = last
                assert h == expected
            else:
                assert h == list(reversed(range(max(0, (i * 2) - 4), i * 2)))


def test_without_updating_history():
    users = ["user_1", "user_"]


def test_non_increasing_timestamps():
    pass


def test_time_lag():
    pass
