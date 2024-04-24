import random
import pytest
from thirdai import data
import numpy as np

pytestmark = [pytest.mark.unit]


def categorical_temporal(include_current_row, should_update_history=True, time_lag=0):
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
        time_lag=time_lag,
    )


def make_column_map(users, values, timestamps):
    return data.ColumnMap(
        {
            "users": data.columns.StringColumn(users),
            "values": data.columns.DecimalColumn(values),
            "timestamps": data.columns.TimestampColumn(timestamps),
        }
    )


def test_numerical_temporal():
    n_users = 3
    occurrences_per_user = 8

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
    columns = categorical_temporal(include_current_row=True)(columns)
    outputs = columns["history"].data()

    for i, output in enumerate(outputs):
        user = i % n_users
        user_occurrence = i // n_users

        print(f"User={user}, occurrence={user_occurrence}")

        start = user * occurrences_per_user
        user_counts = [0.0] * 8 + list(range(start, start + user_occurrence + 1))

        if len(user_counts) % 2 == 1:
            user_counts.append(0)

        interval_counts = np.sum(np.array(user_counts[-8:]).reshape((4, 2)), axis=1)

        if np.sum(interval_counts) > 0:
            interval_counts = interval_counts - np.mean(interval_counts)
            norm = np.linalg.norm(interval_counts, ord=2)
            if norm > 0:
                interval_counts /= norm

        print(output)
        print(interval_counts)
        assert np.allclose(np.array(output), interval_counts)


test_numerical_temporal()
