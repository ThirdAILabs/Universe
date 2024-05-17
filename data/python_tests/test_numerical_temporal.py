import numpy as np
import pytest
from thirdai import data

pytestmark = [pytest.mark.unit]


HISTORY_LEN = 4
INTERVAL_LEN = 2


def numerical_temporal(include_current_row, should_update_history=True, interval_lag=0):
    return data.transformations.NumericalTemporal(
        user_column="users",
        value_column="values",
        timestamp_column="timestamps",
        output_column="history",
        tracker_key="users_to_items",
        history_len=HISTORY_LEN,
        interval_len=INTERVAL_LEN,
        include_current_row=include_current_row,
        should_update_history=should_update_history,
        interval_lag=interval_lag,
    )


def make_column_map(users, values, timestamps):
    return data.ColumnMap(
        {
            "users": data.columns.StringColumn(users),
            "values": data.columns.DecimalColumn(values),
            "timestamps": data.columns.TimestampColumn(timestamps),
        }
    )


def normalize_counts(counts):
    counts = counts - np.mean(counts)
    norm = np.linalg.norm(counts, ord=2)
    if norm > 0:
        counts /= norm
    return counts


@pytest.mark.parametrize(
    "include_current_row,interval_lag", [(True, 0), (False, 0), (False, 3)]
)
def test_numerical_temporal(include_current_row, interval_lag):
    """
    Say we have 3 users and 4 occurrences per user. Then this test creates a dataset
    which looks like this:

    +-----------+--------+--------+--------+
    | timestamp | user 0 | user 1 | user 2 |
    +-----------+--------+--------+--------+
    |         0 |      0 |      4 |      8 |
    |         1 |      1 |      5 |      9 |
    |         2 |      2 |      6 |     10 |
    |       ... |    ... |    ... |    ... |
    +-----------+--------+--------+--------+

    """
    n_users = 10
    occurrences_per_user = 12

    user_ids = []
    values = []
    timestamps = []

    for i in range(occurrences_per_user):
        user_ids.extend(map(str, range(n_users)))
        values.extend(
            range(i, n_users * occurrences_per_user + i, occurrences_per_user)
        )
        timestamps.extend([i] * n_users)

    columns = make_column_map(users=user_ids, values=values, timestamps=timestamps)
    columns = numerical_temporal(
        include_current_row=include_current_row, interval_lag=interval_lag
    )(columns)
    outputs = columns["history"].data()

    for i, output in enumerate(outputs):
        # The data is ordered:
        # (time 0, user 0), (time 0, user 1), ... (time 1, user 0, time 1, user 1), ...
        user = i % n_users
        timestep = i // n_users

        # This creates an array that looks like this (using user 0 for an example):
        # [[0, 1],
        #  [2, 3],
        #  [4, 5], ... ]
        # The reason it is grouped into pairs is the the interval length is 2 so
        # each row represents a bucket that the temporal transformation will consider.
        full_user_counts = np.arange(
            user * occurrences_per_user, (user + 1) * occurrences_per_user
        ).reshape((-1, INTERVAL_LEN))

        # This gives the the most recent interval we could possibly consider for
        # the current sample, this is adjusted later to account for how far into
        # the interval the current timestamp is, or if we are not including the current row.
        last_interval_to_use = max(1 + timestep // INTERVAL_LEN - interval_lag, 0)
        full_user_counts = full_user_counts[:last_interval_to_use]

        if interval_lag == 0:
            if timestep % INTERVAL_LEN == 0:
                # If we're at an even timestep, and thus the first timestep in the
                # interval, zero out the count of the second timestep in the interval.
                full_user_counts[-1, -1] = 0
            if not include_current_row:
                # If we're not including the current timestep's count, zero it out too.
                full_user_counts[-1, timestep % INTERVAL_LEN] = 0

        # We add padding here so that we can just take the last HISTORY_LEN intervals.
        full_user_counts = np.concatenate(
            [np.zeros((HISTORY_LEN, INTERVAL_LEN)), full_user_counts], axis=0
        )
        full_user_counts = full_user_counts[-HISTORY_LEN:]

        interval_counts = full_user_counts.sum(axis=-1)

        interval_counts = normalize_counts(interval_counts)

        assert np.allclose(np.array(output), interval_counts)


def normalize_and_compare(counts, expected_counts):
    assert len(counts) == len(expected_counts)
    for c, ec in zip(counts, expected_counts):
        assert np.allclose(np.array(c), normalize_counts(np.array(ec)))


def test_non_updating_transform():
    columns = make_column_map(
        users=["2", "1", "1", "2"], timestamps=[2, 1, 5, 6], values=[1, 2, 3, 4]
    )

    state = data.transformations.State()

    columns = numerical_temporal(True)(columns, state)

    new_columns = make_column_map(
        users=["1", "2", "1", "2"], timestamps=[7, 6, 9, 8], values=[5, 6, 7, 8]
    )

    w_curr_row = numerical_temporal(True, should_update_history=False)(
        new_columns, state
    )
    normalize_and_compare(
        counts=w_curr_row["history"].data(),
        # Full histories of users as tuples of (timestamp, value):
        # [User 1] from columns: [(1, 2), (5, 3)] from new_columns: [(7, 5), (9, 7)]
        # [User 2] from_columns: [(2, 1), (6, 4)] from new_columns: [(6, 6), (8, 8)]
        expected_counts=[
            # User 1, cur timestamp 7. Intervals at each column: 0-1, 2-3, 4-5, 6-7
            [2.0, 0.0, 3.0, 5.0],
            # User 2, cur timestamp 6. Intervals: 0-1, 2-3, 4-5, 6-7
            [0.0, 1.0, 0.0, 10.0],
            # User 1, timestamp 9. Intervals: 2-3, 4-5, 6-7, 8-9
            # 6-7 interval is 0 because should_update_history = False
            [0.0, 3.0, 0.0, 7.0],
            # User 2, timestamp 8. Intervals: 2-3, 4-5, 6-7, 8-9
            # 6-7 interval is 4 because should_update_history = False
            [1.0, 0.0, 4.0, 8.0],
        ],
    )

    wo_curr_row = numerical_temporal(False, should_update_history=False)(
        new_columns, state
    )
    normalize_and_compare(
        counts=wo_curr_row["history"].data(),
        # Since include_current_row and should_update_history are False,
        # The values below do not reflect counts from `new_columns`.
        # It only uses `new_columns` for timestamps.
        expected_counts=[
            # User 1, timestamp 7. Intervals: 0-1, 2-3, 4-5, 6-7
            [2.0, 0.0, 3.0, 0.0],
            # User 2, cur timestamp 6. Intervals: 0-1, 2-3, 4-5, 6-7
            [0.0, 1.0, 0.0, 4.0],
            # User 1, timestamp 9. Intervals: 2-3, 4-5, 6-7, 8-9
            [0.0, 3.0, 0.0, 0.0],
            # User 2, timestamp 8. Intervals: 2-3, 4-5, 6-7, 8-9
            [1.0, 0.0, 4.0, 0.0],
        ],
    )


def test_non_increasing_timestamps():
    users = ["user_1", "user_2", "user_2", "user_1"]
    values = [1.0, 2.0, 3.0, 4.0]
    timestamps = [4, 2, 3, 1]

    columns = make_column_map(users, values, timestamps)

    transformation = numerical_temporal(include_current_row=True)

    with pytest.raises(
        ValueError,
        match="Expected increasing timestamps for each tracking key. Found timestamp in the interval \[0, 2\) after seeing timestamp in the interval \[4, 6\) for tracking key 'user_1'.",
    ):
        transformation(columns)
