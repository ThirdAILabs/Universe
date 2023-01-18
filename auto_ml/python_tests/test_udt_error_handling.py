import os

import pytest
from thirdai import bolt

pytestmark = [pytest.mark.unit]


def create_simple_file(col_names, num_data_cols, file_name, num_rows=5):
    with open(file_name, "w") as f:
        f.write(",".join(col_names) + "\n")
        for _ in range(num_rows):
            f.write(",".join(["1" for _ in range(num_data_cols)]) + "\n")


def test_too_many_cols_in_train():
    """
    This test expect an error because this dataset has 3 columns in
    the header but has 4 columns in subsequent rows (UDT expects the same number
    of columns in every row).
    """
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
        match='Expected 3 columns in each row of the dataset. Found row with 4 columns: "1" "1" "1" "1"',
    ):
        model.train("too_many_cols", epochs=100)

    os.remove("too_many_cols")


def test_too_few_cols_in_train():
    """
    This test expect an error because this dataset has 3 columns in
    the header but has 2 columns in subsequent rows (UDT expects the same number
    of columns in every row).
    """
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
        match='Expected 3 columns in each row of the dataset. Found row with 2 columns: "1" "1"',
    ):
        model.train("too_few_cols", epochs=100)

    os.remove("too_few_cols")


def test_header_missing_cols():
    create_simple_file(
        col_names=["a", "b"], num_data_cols=2, file_name="header_missing_cols"
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
        RuntimeError,
        match="Expected a column named 'c' in header but could not find it",
    ):
        model.train("header_missing_cols", epochs=100)

    os.remove("header_missing_cols")


def test_target_not_in_data_types():
    with pytest.raises(
        ValueError,
        match="Target column provided was not found in data_types.",
    ):
        bolt.UniversalDeepTransformer(
            data_types={
                "text_col": bolt.types.text(),
                "some_random_name": bolt.types.categorical(),
            },
            target="target",
            n_target_classes=2,
        )


def test_contextual_text_encodings():
    invalid_encoding = "INVALID"
    with pytest.raises(
        ValueError,
        match=f"Created text column with invalid contextual_encoding '{invalid_encoding}' please choose one of 'none', 'local', or 'global'."
    ):
        bolt.UniversalDeepTransformer(
            data_types={
                "text_col": bolt.types.text(contextual_encoding=invalid_encoding),
                "some_random_name": bolt.types.categorical(),
            },
            target="target",
            n_target_classes=2,
        )
