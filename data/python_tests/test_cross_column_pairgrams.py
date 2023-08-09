import pytest
from dataset_utils import verify_hash_distribution
from thirdai import data

pytestmark = [pytest.mark.unit]


def test_cross_column_pairgrams_consistency():
    N = 20

    """
    This creates a column map that looks like this (this is if N=10 to save space):

    [ 0  1  2  3  4  5  6  7  8  9]
    [ 0  1  2  3  4  5  6  7  8 11]
    [ 0  1  2  3  4  5  6  7 11 12]
    [ 0  1  2  3  4  5  6 11 12 13]
    [ 0  1  2  3  4  5 11 12 13 14]
    [ 0  1  2  3  4 11 12 13 14 15]
    [ 0  1  2  3 11 12 13 14 15 16]
    [ 0  1  2 11 12 13 14 15 16 17]
    [ 0  1 11 12 13 14 15 16 17 18]
    [ 0 11 12 13 14 15 16 17 18 19]

    In each row 1 column is changed from the previous row. This pattern means that
    there is an exact number of pairgrams that should overlap between any two rows. 

    For rows i, j (j < i) we have the expected number of pairgrams that overlap 
    to be (N - i) * (N - i + 1) / 2. Note that (N - i) is the number of elements that 
    are unchanged from the first row. 
    """

    columns = data.ColumnMap(
        {
            str(i): data.columns.TokenColumn(
                [i] * (N - i) + list(range(N + 1, N + i + 1))
            )
            for i in range(N)
        }
    )

    featurizer = data.transformations.CrossColumnPairgrams(
        input_columns=[str(i) for i in range(N)],
        output_column="pairgrams",
        hash_range=100000000,
    )

    columns = featurizer(columns)

    pairgrams = columns["pairgrams"].data()
    pairgram_sets = [set(x) for x in pairgrams]

    n_pairgrams = N * (N + 1) / 2

    for i in range(len(pairgrams)):
        assert len(pairgrams[i]) == n_pairgrams

        for j in range(i):
            overlap = (N - i) * (N - i + 1) / 2
            assert len(pairgram_sets[j].intersection(pairgram_sets[i])) == overlap


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
