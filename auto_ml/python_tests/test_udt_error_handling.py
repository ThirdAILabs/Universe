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
            "c": bolt.types.categorical(n_classes=2),
        },
        target="c",
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
            "c": bolt.types.categorical(n_classes=2),
        },
        target="c",
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
            "c": bolt.types.categorical(n_classes=2),
        },
        target="c",
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
        match="Target column 'target' not found in data types.",
    ):
        bolt.UniversalDeepTransformer(
            data_types={
                "text_col": bolt.types.text(),
                "some_random_name": bolt.types.categorical(n_classes=2),
            },
            target="target",
        )


@pytest.mark.parametrize("mach", [True, False])
def test_invalid_column_name_in_udt_predict(mach):
    model = bolt.UniversalDeepTransformer(
        data_types={
            "text_col": bolt.types.text(contextual_encoding="local"),
            "target": bolt.types.categorical(n_classes=10, type="int"),
        },
        target="target",
        extreme_classification=mach,
    )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Unable to find column with name 'text_col'. ColumnMap contains columns ['HAHAHA']."
        ),
    ):
        model.predict({"HAHAHA": "some text"})
        model.predict_batch([{"HAHAHA": "some text"}])


@pytest.mark.unit
def test_set_output_sparsity_throws_error_on_unsupported_backend():
    """
    set_output_sparsity is enabled only for UDTClassifier Backend hence, this should throw an error.
    """
    model = bolt.UniversalDeepTransformer(
        data_types={"source": bolt.types.text(), "target": bolt.types.text()},
        target="target",
    )

    with pytest.raises(
        RuntimeError, match=re.escape(f"Method not supported for the model")
    ):
        model.set_output_sparsity(sparsity=0.2, rebuild_hash_tables=False)


def test_invalid_cast_string_to_int():
    with open("train.csv", "w") as f:
        f.write("text,id\n")
        f.write("abcd,ab ef gh")

    model = bolt.UniversalDeepTransformer(
        data_types={
            "text": bolt.types.text(),
            "id": bolt.types.categorical(n_classes=2, type="str"),
        },
        target="id",
    )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Cannot assign id to a new string 'gfh'. The buffer has reached its maximum size of 2."
        ),
    ):
        model.train("train.csv")


def test_invalid_target_column_type():
    with pytest.raises(
        ValueError,
        match=re.escape(
            """Target data type node_id is not valid for a UniversalDeepTransformer model with graph classification.
The following target types are supported to initialize a UniversalDeepTransformer:"""
        ),
    ):
        model = bolt.UniversalDeepTransformer(
            data_types={
                "node_id": bolt.types.node_id(),
                "neighbors": bolt.types.neighbors(),
            },
            target="node_id",
        )
