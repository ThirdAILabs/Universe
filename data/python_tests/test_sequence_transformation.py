from thirdai import data
import pytest
from dataset_utils import verify_hash_distribution

pytestmark = [pytest.mark.unit]


def make_sequence_column(rows, cols):
    return data.ColumnMap(
        {
            "sequence": data.columns.StringColumn(
                [
                    "-".join(map(str, range(i * cols, (i + 1) * cols)))
                    for i in range(rows)
                ]
            )
        }
    )


def test_sequence_hash_distribution():
    ROWS = 400
    COLS = 10
    OUTPUT_RANGE = 100
    columns = make_sequence_column(ROWS, COLS)

    transformation = data.transformations.Sequence(
        "sequence", "tokens", "-", dim=OUTPUT_RANGE
    )

    columns = transformation(columns)

    verify_hash_distribution(columns["tokens"].data(), OUTPUT_RANGE)


def test_sequence_hash_consistency():
    ROWS = 100
    COLS = 10

    columns = make_sequence_column(ROWS, COLS)

    transformation = data.transformations.Sequence("sequence", "tokens", "-")

    columns = transformation(columns)

    tokens = columns["tokens"].data()
    for i in range(len(tokens) - 1):
        row = set(tokens[i])
        assert len(row) == COLS
        assert len(row.intersection(set(tokens[i + 1]))) == 0


def test_sequence_hash_overlap():
    N = 20

    columnns = data.ColumnMap(
        {
            "sequence": data.columns.StringColumn(
                [
                    "-".join(map(str, list(range(i + 1)) + [i + N] * (N - i - 1)))
                    for i in range(N)
                ]
            )
        }
    )

    transformation = data.transformations.Sequence("sequence", "tokens", "-")

    columns = transformation(columnns)

    rows = [set(x) for x in columns["tokens"].data()]

    for i in range(N):
        for j in range(i):
            assert len(rows[i].intersection(rows[j])) == (j + 1)
