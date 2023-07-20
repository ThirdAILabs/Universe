import pytest
from thirdai import data, dataset


@pytest.mark.unit
def test_mach_transformation():
    columns = data.ColumnMap(
        {"ids": data.columns.TokenArrayColumn([[0, 2], [1], [0, 1, 3]])}
    )

    index = dataset.MachIndex(
        entity_to_hashes={0: [4, 7, 2], 1: [8, 0, 3], 2: [7, 1, 5], 3: [9, 6, 0]},
        num_hashes=3,
        output_range=10,
    )

    transformation = data.transformations.MachLabel(
        input_column="ids",
        output_column="hashes",
        index=index,
    )

    columns = transformation(columns)

    expected_hashes = [[4, 7, 2, 7, 1, 5], [8, 0, 3], [4, 7, 2, 8, 0, 3, 9, 6, 0]]

    assert columns["hashes"].data() == expected_hashes
