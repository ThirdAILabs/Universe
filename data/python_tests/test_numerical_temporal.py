import numpy as np
import pytest
from thirdai import data

pytestmark = [pytest.mark.unit]


def numerical_temporal(include_current_row, should_update_history=True, interval_lag=0):
    return data.transformations.NumericalTemporal(
        user_column="users",
        value_column="values",
        timestamp_column="timestamps",
        output_column="history",
        tracker_key="users_to_items",
        history_len=4,
        interval_len=2,
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
    n_users = 1
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
        user = i % n_users
        step = i // n_users

        full_user_counts = np.arange(
            user * occurrences_per_user, (user + 1) * occurrences_per_user
        ).reshape((-1, 2))

        last_interval_to_use = max(1 + step // 2 - interval_lag, 0)
        full_user_counts = full_user_counts[:last_interval_to_use]

        if interval_lag == 0:
            if step % 2 == 0:
                # If we're at an even step, and thus the first count in the
                # interval, zero out the second count in the interval.
                full_user_counts[-1, -1] = 0
            if not include_current_row:
                # If we're not including the current step's count, zero it out too.
                full_user_counts[-1, step % 2] = 0

        full_user_counts = np.concatenate([np.zeros((4, 2)), full_user_counts], axis=0)
        full_user_counts = full_user_counts[-4:]

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
        expected_counts=[
            [2.0, 0.0, 3.0, 5.0],
            [0.0, 1.0, 0.0, 10.0],
            [0.0, 3.0, 0.0, 7.0],
            [1.0, 0.0, 4.0, 8.0],
        ],
    )

    wo_curr_row = numerical_temporal(False, should_update_history=False)(
        new_columns, state
    )
    normalize_and_compare(
        counts=wo_curr_row["history"].data(),
        expected_counts=[
            [2.0, 0.0, 3.0, 0.0],
            [0.0, 1.0, 0.0, 4.0],
            [0.0, 3.0, 0.0, 0.0],
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
