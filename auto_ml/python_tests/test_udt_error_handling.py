import pytest
from thirdai import bolt

pytestmark = [pytest.mark.unit]


def create_simple_file(col_names, num_data_cols, file_name, num_rows=5):
    with open(file_name, "w") as f:
        f.write(",".join(col_names) + "\n")
        for _ in range(num_rows):
            f.write(",".join(["1" for _ in range(num_data_cols)]) + "\n")


def test_too_many_cols_in_train():
    create_simple_file(
        col_names=["a", "b", "c"], num_data_cols=4, file_name="too_many_cols"
    )

    model = bolt.UniversalDeepTransformer(
        data_types={
            "a": bolt.types.text(),
            "b": bolt.types.text(),
            "c": bolt.types.categorical(),
        },
        target="c",
        n_target_classes=2,
    )

    with pytest.raises(
        ValueError,
        match=".*Expected 3 columns delimited by ',' in each row of the dataset. Found row '1,1,1,1' with number of columns = 4.",
    ):
        model.train("too_many_cols", epochs=100)


def test_too_few_cols_in_train():
    create_simple_file(
        col_names=["a", "b", "c"], num_data_cols=2, file_name="too_few_cols"
    )

    model = bolt.UniversalDeepTransformer(
        data_types={
            "a": bolt.types.text(),
            "b": bolt.types.text(),
            "c": bolt.types.categorical(),
        },
        target="c",
        n_target_classes=2,
    )

    with pytest.raises(
        ValueError,
        match=".*Expected 3 columns delimited by ',' in each row of the dataset. Found row '1,1' with number of columns = 2.",
    ):
        model.train("too_few_cols", epochs=100)
