import random
from collections import defaultdict

import pytest
from thirdai import data

pytestmark = [pytest.mark.unit]


def categorical_temporal(include_current_row, should_update_history=True, time_lag=0):
    return data.transformations.CategoricalTemporal(
        user_column="users",
        item_column="items",
        timestamp_column="timestamps",
        output_column="history",
        tracker_key="users_to_items",
        track_last_n=4,
        include_current_row=include_current_row,
        should_update_history=should_update_history,
        time_lag=time_lag,
    )


def make_column_map(users, items, timestamps):
    return data.ColumnMap(
        {
            "users": data.columns.StringColumn(users),
            "items": data.columns.TokenArrayColumn(items),
            "timestamps": data.columns.TimestampColumn(timestamps),
        }
    )


@pytest.mark.parametrize("include_current_row", [True, False])
def test_categorical_temporal_ascending_item_ids(include_current_row):
    N_USERS = 20
    occurrences_PER_USER = 10
    users = [str(i) for i in range(N_USERS)] * occurrences_PER_USER
    random.shuffle(users)
    timestamps = list(range(len(users)))
    items = []
    user_occurrences = defaultdict(int)
    for user in users:
        item_id = int(user) * occurrences_PER_USER * 2 + user_occurrences[user]
        items.append([item_id, item_id + 1])
        user_occurrences[user] += 2

    columns = make_column_map(users, items, timestamps)

    transformation = categorical_temporal(include_current_row)

    columns = transformation(columns)

    user_histories = defaultdict(list)

    for i in range(len(columns)):
        user_histories[columns["users"][i]].append(list(columns["history"][i]))

    assert len(user_histories) == N_USERS

    for user, histories in user_histories.items():
        assert len(histories) == occurrences_PER_USER

        # Because we group the samples by user, and the timestamps are strictly
        # increasing we can make assertions about what items are in each sample.
        for i, h in enumerate(histories):
            user_start = int(user) * occurrences_PER_USER * 2
            curr_start = user_start + i * 2
            if include_current_row:
                expected = list(
                    reversed(range(max(user_start, curr_start - 2), curr_start + 2))
                )
            else:
                expected = list(
                    reversed(range(max(user_start, curr_start - 4), curr_start))
                )
            assert h == expected


@pytest.mark.parametrize("include_current_row", [True, False])
def test_without_updating_history(include_current_row):
    users = ["user_1", "user_2", "user_2", "user_1"]
    items = [[0, 1], [10, 11], [12, 13], [2, 3]]
    timestamps = [0, 1, 2, 3]

    columns = make_column_map(users, items, timestamps)

    updating_transformation = categorical_temporal(include_current_row)

    nonupdating_transformation = categorical_temporal(
        include_current_row, should_update_history=False
    )

    state = data.transformations.State()

    updating_transformation(columns, state)

    users = ["user_2", "user_1", "user_1", "user_2"]
    items = [[14, 15], [4, 5], [6, 7], [16, 17]]
    timestamps = [10, 11, 12, 13]

    columns = make_column_map(users, items, timestamps)

    # Here we invoke the non updating transformation on a new list of samples.
    # The same users occur multiple times in this new list of samples. This tests
    # the non-updating property because we check that these featurized samples
    # only contain history from the samples indexed with the updating transformation.
    columns = nonupdating_transformation(columns, state)

    if include_current_row:
        expected_rows = [[12, 13, 14, 15], [2, 3, 4, 5], [2, 3, 6, 7], [12, 13, 16, 17]]
    else:
        expected_rows = [[10, 11, 12, 13], [0, 1, 2, 3], [0, 1, 2, 3], [10, 11, 12, 13]]
    for row, expected_row in zip(columns["history"].data(), expected_rows):
        assert set(row) == set(expected_row)


def test_non_increasing_timestamps():
    users = ["user_1", "user_2", "user_2", "user_1"]
    items = [[0, 1], [10, 11], [12, 13], [2, 3]]
    timestamps = [4, 1, 2, 3]

    columns = make_column_map(users, items, timestamps)

    transformation = categorical_temporal(include_current_row=True)

    with pytest.raises(
        ValueError,
        match="Expected increasing timestamps for each tracking key. Found timestamp 3 after seeing timestamp 4 for tracking key 'user_1'.",
    ):
        transformation(columns)


def test_time_lag():
    users = ["user_1", "user_2", "user_2", "user_1", "user_2", "user_1", "user_2"]
    items = [[0, 1], [10, 11], [12, 13], [2, 3], [14, 15], [4, 5], [16, 17]]
    timestamps = [0, 0, 1, 1, 2, 2, 3]

    columns = make_column_map(users, items, timestamps)

    transformation = categorical_temporal(include_current_row=True, time_lag=2)

    columns = transformation(columns)

    expected_rows = [
        [],  # First items for user_1.
        [],  # First items for user_2.
        [],  # Second items for user_2, but time lag rules out first.
        [],  # Second items for user_1, but time lag rules out first.
        [10, 11],  # Third items for user_2, but time lag rules out second.
        [0, 1],  # Third items for user_1,  but time lag rules out second.
        [10, 11, 12, 13],  # Fourth items for user_2, but time lag rules out third.
    ]

    for row, expected_row in zip(columns["history"].data(), expected_rows):
        assert set(row) == set(expected_row)
