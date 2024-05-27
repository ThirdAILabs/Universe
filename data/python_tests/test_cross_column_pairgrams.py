import pytest
from dataset_utils import verify_hash_distribution
from thirdai import data

pytestmark = [pytest.mark.unit]


def get_column_map(N):
    return data.ColumnMap(
        {
            str(i): data.columns.TokenColumn(
                [i] * (N - i) + list(range(N + 1, N + i + 1))
            )
            for i in range(N)
        }
    )


def get_transform(N):
    return data.transformations.CrossColumnPairgrams(
        input_columns=[str(i) for i in range(N)],
        output_column="pairgrams",
        hash_range=100000000,
    )


def test_cross_column_pairgrams_consistency():
    N = 20

    """
    This creates a column map that looks like this (this is if N=8 to save space):

    Col 0 | Col 1 | Col 2 | Col 3 | Col 4 | Col 5 | Col 6 | Col 7
      0   |   1   |   2   |   3   |   4   |   5   |   6   |   7 
      0   |   1   |   2   |   3   |   4   |   5   |   6   |   9 
      0   |   1   |   2   |   3   |   4   |   5   |   9   |  10 
      0   |   1   |   2   |   3   |   4   |   9   |  10   |  11 
      0   |   1   |   2   |   3   |   9   |  10   |  11   |  12 
      0   |   1   |   2   |   9   |  10   |  11   |  12   |  13 
      0   |   1   |   9   |  10   |  11   |  12   |  13   |  14 
      0   |   9   |  10   |  11   |  12   |  13   |  14   |  15 


    In each row 1 column is changed from the previous row. This pattern means that
    there is an exact number of pairgrams that should overlap between any two rows. 

    For rows i, j (with j < i), the number of columns whose values match is (N - i)
    since (N - i) is the number of elements that are unchanged from the first row.
    Thus the expected number of pairgrams that overlap is (N - i) * (N - i + 1) / 2, 
    since M * (M + 1) / 2 is the number of pairgrams that are created from M tokens.

    Additionally, the non overlapping elements between two rows have the same values, 
    but in different columns. This also tests that the same values in different 
    columns don't overlap.
    """

    columns = get_column_map(N)

    transform = get_transform(N)

    columns = transform(columns)

    pairgrams = columns["pairgrams"].data()
    pairgram_sets = [set(x) for x in pairgrams]

    n_pairgrams = N * (N + 1) / 2

    for i in range(len(pairgrams)):
        assert len(pairgrams[i]) == n_pairgrams

        for j in range(i):
            overlap = (N - i) * (N - i + 1) / 2
            assert len(pairgram_sets[j].intersection(pairgram_sets[i])) == overlap


def test_cross_column_pairgram_serialization():
    N = 10

    columns = get_column_map(N)

    transform = get_transform(N)

    original_output = transform(columns)["pairgrams"].data()

    new_transform = data.transformations.deserialize(transform.serialize())

    new_output = new_transform(columns)["pairgrams"].data()

    assert original_output == new_output


def test_cross_column_pairgram_hash_distribution():
    ROWS = 1000
    COLS = 10
    HASH_RANGE = 1000

    columns = data.ColumnMap(
        {
            str(i): data.columns.TokenColumn(list(range(i, i + ROWS)))
            for i in range(COLS)
        }
    )

    featurizer = data.transformations.CrossColumnPairgrams(
        input_columns=[str(i) for i in range(COLS)],
        output_column="pairgrams",
        hash_range=HASH_RANGE,
    )

    columns = featurizer(columns)

    verify_hash_distribution(columns["pairgrams"], HASH_RANGE)
