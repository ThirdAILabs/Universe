import os
import re

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
        match="Expected 3 columns. But received row '1,1,1,1' with 4 columns.",
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
        ValueError, match="Expected 3 columns. But received row '1,1' with 2 columns."
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

    # TODO(Nicholas): Is it ok to display the intermediate column names?
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Unable to find column with name 'c'. ColumnMap contains columns ['__featurized_input_values__', '__b_tokenized__', '__featurized_input_indices__', '__a_tokenized__', 'b', 'a']."
        ),
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


@pytest.mark.parametrize("mach", [True, False])
def test_invalid_column_name_in_udt_predict(mach):
    model = bolt.UniversalDeepTransformer(
        data_types={
            "text_col": bolt.types.text(contextual_encoding="local"),
            "target": bolt.types.categorical(),
        },
        target="target",
        n_target_classes=10,
        integer_target=True,
        options={
            "extreme_classification": mach,
        },
    )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Unable to find column with name 'text_col'. ColumnMap contains columns ['HAHAHA']."
        ),
    ):
        model.predict({"HAHAHA": "some text"})
        model.predict_batch([{"HAHAHA": "some text"}])
