import pytest
from thirdai import data, dataset


@pytest.mark.parametrize("serialize", [True, False])
@pytest.mark.unit
def test_mach_transformation(serialize):
    columns = data.ColumnMap(
        {"ids": data.columns.TokenArrayColumn([[0, 2], [1], [0, 1, 3]])}
    )

    transformation = data.transformations.MachLabel(
        input_column="ids",
        output_column="hashes",
    )
    if serialize:
        transformation = data.transformations.deserialize(transformation.serialize())

    index = dataset.MachIndex(
        entity_to_hashes={0: [4, 7, 2], 1: [8, 0, 3], 2: [7, 1, 5], 3: [9, 6, 0]},
        num_hashes=3,
        output_range=10,
    )

    columns = transformation(columns, state=data.transformations.State(index))

    expected_hashes = [[4, 7, 2, 7, 1, 5], [8, 0, 3], [4, 7, 2, 8, 0, 3, 9, 6, 0]]

    assert columns["hashes"].data() == expected_hashes


@pytest.mark.unit
def test_mach_transformation_error_handling():
    columns = data.ColumnMap({"ids": data.columns.TokenColumn([0, 1])})

    transformation = data.transformations.MachLabel(
        input_column="ids",
        output_column="hashes",
    )

    with pytest.raises(
        ValueError, match="Transformation state does not contain MachIndex."
    ):
        transformation(columns)

    index = dataset.MachIndex(entity_to_hashes={0: [0]}, num_hashes=1, output_range=10)

    with pytest.raises(ValueError, match="Invalid entity in index: 1."):
        transformation(columns, state=data.transformations.State(index))
