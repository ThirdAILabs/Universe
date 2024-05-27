import pytest
from thirdai import data

pytestmark = [pytest.mark.unit]


def get_str_col(col_length):
    return data.columns.StringColumn([f"value{i}" for i in range(col_length)])


def get_two_col_hashed_string_dataset(col_length, output_range):
    column1, column2 = get_str_col(col_length), get_str_col(col_length)

    columns = data.ColumnMap({"column1": column1, "column2": column2})

    featurizer = data.transformations.Pipeline(
        transformations=[
            data.transformations.StringHash(
                input_column=column_name,
                output_column=f"{column_name}_hashes",
                output_range=output_range,
            )
            for column_name in ["column1", "column2"]
        ]
    )

    columns = featurizer(columns)

    return list(zip(columns["column1_hashes"].data(), columns["column2_hashes"].data()))


# Tests that if we hash two columns and then turn them into a dataset, the sparse
# indices module the output range of the first column will be the same (this
# ensures that the hash function is consistent).
def test_string_hash_consistency():
    col_length = 100
    output_range = 100

    indices = get_two_col_hashed_string_dataset(col_length, output_range)

    for i1, i2 in indices:
        assert i1 == i2


# Tests that each hash has about the number of values we expect (is within a
# factor of 2 of the expected count). This won't be flaky because the hash is
# seeded and thus entirely deterministic.
def test_string_hash_distribution():
    col_length = 10000
    output_range = 100

    indices = get_two_col_hashed_string_dataset(col_length, output_range)

    hash_counts = [0 for _ in range(output_range)]
    for i1, _ in indices:
        hash_counts[i1] += 1

    expected_count = col_length / output_range
    for count in hash_counts:
        assert count / expected_count < 2 and count / expected_count > 0.5


def test_string_hash_with_delimiter():
    ROWS = 100
    columns = data.ColumnMap(
        {
            "col": data.columns.StringColumn(
                [f"{i}-{i+1}-{i}-{i+1}" for i in range(ROWS)]
            )
        }
    )

    str_hash = data.transformations.StringHash(
        input_column="col",
        output_column=f"hashes",
        delimiter="-",
        output_range=10000,
    )

    columns = str_hash(columns)

    hashes = columns["hashes"].data()

    for i, row in enumerate(hashes):
        assert row[0] == row[2]
        assert row[1] == row[3]
        assert row[0] != row[1]

        if i < len(hashes) - 1:
            assert row[1] == hashes[i + 1][0]


def test_string_hash_serialization():
    N = 20
    columns = data.ColumnMap(
        {"str": data.columns.StringColumn([f"val_{i % N}" for i in range(2 * N)])}
    )

    transformation = data.transformations.StringHash("str", "hashes", output_range=1000)

    transformation_copy = data.transformations.deserialize(transformation.serialize())

    output1 = transformation(columns)
    output2 = transformation_copy(columns)

    assert output1["hashes"].data() == output2["hashes"].data()
    assert output2["hashes"].data()[:N] == output2["hashes"].data()[N:]
