import re

import numpy as np
import pytest
from thirdai import bolt

SIMPLE_VOCAB_SIZE = 1000
SIMPLE_METADATA_DIM = 100
SIMPLE_N_CLASSES = 10


def get_input_data(batch_size):
    tokens_per_sample = 20
    return {
        "tokens": np.random.randint(
            0, SIMPLE_VOCAB_SIZE, size=batch_size * tokens_per_sample, dtype=np.uint32
        ),
        "offsets": np.arange(
            start=0,
            stop=batch_size * tokens_per_sample + 1,
            step=tokens_per_sample,
            dtype=np.uint32,
        ),
        "metadata": np.random.randint(
            0, 2, size=(batch_size, SIMPLE_METADATA_DIM), dtype=np.uint32
        ),
    }


def get_labels(batch_size):
    return np.random.randint(0, 2, size=(batch_size, SIMPLE_N_CLASSES)).astype(
        np.float32
    )


def check_model_operations(
    input_data, labels, error_msg, error_type=ValueError, input_data_corrupted=True
):
    model = bolt.UniversalDeepTransformer(
        input_vocab_size=SIMPLE_VOCAB_SIZE,
        metadata_dim=SIMPLE_METADATA_DIM,
        n_classes=SIMPLE_N_CLASSES,
        model_size="small",
    )
    with pytest.raises(error_type, match=error_msg):
        model.train(input_data, labels, 0.1)
    with pytest.raises(error_type, match=error_msg):
        model.validate(input_data, labels)

    if input_data_corrupted:
        with pytest.raises(error_type, match=error_msg):
            model.predict(input_data)
    else:
        model.predict(input_data)


@pytest.mark.parametrize("field", ["tokens", "offsets", "metadata"])
def test_missing_input(field):
    batch_size = 5
    input_data = get_input_data(batch_size)
    del input_data[field]

    check_model_operations(
        input_data=input_data,
        labels=get_labels(batch_size),
        error_msg=re.escape(field),
        error_type=KeyError,
    )


@pytest.mark.parametrize(
    "field, error",
    [  # When we add additional tokens the last offset doesn't match the end of the tokens.
        ("tokens", "The last offset should be the number of tokens + 1."),
        (  # The batch size is infered from the number of offsets.
            "offsets",
            "Expected metadata to have shape (6, 100, ), but received array with shape (5, 100, ).",
        ),
        (  # The batch size is infered from the number of offsets.
            "metadata",
            "Expected metadata to have shape (5, 100, ), but received array with shape (6, 100, ).",
        ),
    ],
)
def test_incorrect_batch_size(field, error):
    batch_size = 5
    input_data = get_input_data(batch_size)

    if field == "offsets":
        # We add an extra valid offset so that the batch size is changed.
        input_data["offsets"] = np.concatenate(
            [np.array([0], dtype=np.uint32), input_data["offsets"]],
            axis=None,
            dtype=np.uint32,
        )
    else:
        bad_input_data = get_input_data(batch_size + 1)
        input_data[field] = bad_input_data[field]

    check_model_operations(
        input_data=input_data, error_msg=re.escape(error), labels=get_labels(batch_size)
    )


@pytest.mark.parametrize(
    "field, error",
    [
        (
            "tokens",
            "Expected tokens to have 1 dimensions, but received array with 3 dimensions.",
        ),
        (
            "offsets",
            "Expected offsets to have 1 dimensions, but received array with 3 dimensions.",
        ),
        (
            "metadata",
            "Expected metadata to have 2 dimensions, but received array with 3 dimensions.",
        ),
    ],
)
def test_incorrect_ndim(field, error):
    batch_size = 5
    input_data = get_input_data(batch_size)

    # The metadata is 2D and the tokens and offsets are both 1D so this array will
    # have invalid dimensions for all inputs.
    input_data[field] = np.random.randint(
        0, 10, size=(batch_size, 100, 10), dtype=np.uint32
    )

    check_model_operations(
        input_data=input_data, labels=get_labels(batch_size), error_msg=re.escape(error)
    )


@pytest.mark.parametrize(
    "field, error",
    [
        (
            "tokens",
            r"We found an Input BoltVector larger than the expected input dim: Received sparse BoltVector with active_neuron=\d+ but was supposed to have=1100",
        ),
        ("offsets", "Invalid offset 1000 for CSR tokens array of length 100."),
        (
            "metadata",
            "Expected metadata to have shape (5, 100, ), but received array with shape (5, 101, ).",
        ),
    ],
)
def test_out_of_range_dimensions(field, error):
    # This test checks how the model handles out of range issues. The test does this
    # by either modifying the indices in the tokens or offsets, or the 2nd dimension
    # of the metadata array.

    batch_size = 5
    input_data = get_input_data(batch_size)
    if field == "metadata":
        input_data["metadata"] = np.random.randint(
            0, 2, size=(batch_size, SIMPLE_METADATA_DIM + 1), dtype=np.uint32
        )
    else:
        input_data[field] += SIMPLE_VOCAB_SIZE

    # We need to match on a number using \d+ so we cannot escape the regex in this case.
    if not field == "tokens":
        error = re.escape(error)

    check_model_operations(
        input_data=input_data, labels=get_labels(batch_size), error_msg=error
    )


def test_incorrect_label_batch_size():
    batch_size = 5
    check_model_operations(
        input_data=get_input_data(batch_size),
        labels=get_labels(batch_size + 1),
        input_data_corrupted=False,
        error_msg=re.escape(
            "Expected labels to have shape (5, 10, ), but received array with shape (6, 10, )."
        ),
    )


def test_incorrect_label_ndim():
    batch_size = 5

    # Labels should be a 2D matrix so this is invalid.
    labels = np.random.randint(0, 2, size=(batch_size, SIMPLE_N_CLASSES, 10)).astype(
        np.float32
    )

    check_model_operations(
        input_data=get_input_data(batch_size),
        labels=labels,
        input_data_corrupted=False,
        error_msg=re.escape(
            "Expected labels to have 2 dimensions, but received array with 3 dimensions."
        ),
    )


def test_incorrect_label_dimension():
    batch_size = 5

    labels = np.random.randint(0, 2, size=(batch_size, SIMPLE_N_CLASSES + 1)).astype(
        np.float32
    )

    check_model_operations(
        input_data=get_input_data(batch_size),
        labels=labels,
        input_data_corrupted=False,
        error_msg=re.escape(
            "Expected labels to have shape (5, 10, ), but received array with shape (5, 11, )."
        ),
    )
