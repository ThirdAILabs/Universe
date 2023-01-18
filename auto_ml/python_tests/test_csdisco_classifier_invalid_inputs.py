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
    input_data, labels, error_type=ValueError, input_data_corrupted=True
):
    model = bolt.UniversalDeepTransformer(
        input_vocab_size=SIMPLE_VOCAB_SIZE,
        metadata_dim=SIMPLE_METADATA_DIM,
        n_classes=SIMPLE_N_CLASSES,
        model_size="small",
    )
    with pytest.raises(error_type):
        model.train(input_data, labels, 0.1)
    with pytest.raises(error_type):
        model.validate(input_data, labels)

    if input_data_corrupted:
        with pytest.raises(error_type):
            model.predict(input_data)
    else:
        model.predict(input_data)


@pytest.mark.parametrize("field", ["tokens", "offsets", "metadata"])
def test_missing_input(field):
    batch_size = 5
    input_data = get_input_data(batch_size)
    del input_data[field]

    check_model_operations(
        input_data=input_data, labels=get_labels(batch_size), error_type=KeyError
    )


@pytest.mark.parametrize("field", ["tokens", "offsets", "metadata"])
def test_incorrect_batch_size(field):
    batch_size = 5
    input_data = get_input_data(batch_size)
    bad_input_data = get_input_data(batch_size + 1)

    input_data[field] = bad_input_data[field]

    check_model_operations(input_data=input_data, labels=get_labels(batch_size))


@pytest.mark.parametrize("field", ["tokens", "offsets", "metadata"])
def test_incorrect_ndim(field):
    batch_size = 5
    input_data = get_input_data(batch_size)

    # The metadata is 2D and the tokens and offsets are both 1D so this array will
    # have invalid dimensions for all inputs.
    input_data[field] = np.random.randint(
        0, 10, size=(batch_size, 100, 10), dtype=np.uint32
    )

    check_model_operations(input_data=input_data, labels=get_labels(batch_size))


@pytest.mark.parametrize("field", ["tokens", "offsets", "metadata"])
def test_out_of_range_dimensions(field):
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

    check_model_operations(input_data=input_data, labels=get_labels(batch_size))


def test_incorrect_label_batch_size():
    batch_size = 5
    check_model_operations(
        input_data=get_input_data(batch_size),
        labels=get_labels(batch_size + 1),
        input_data_corrupted=False,
    )


def test_incorrect_label_ndim():
    batch_size = 5

    # Labels should be a 2D matrix so this is invalid.
    labels = np.random.randint(0, 2, size=(batch_size, SIMPLE_N_CLASSES, 10)).astype(
        np.float32
    )

    check_model_operations(
        input_data=get_input_data(batch_size), labels=labels, input_data_corrupted=False
    )


def test_incorrect_label_dimension():
    batch_size = 5

    labels = np.random.randint(0, 2, size=(batch_size, SIMPLE_N_CLASSES + 1)).astype(
        np.float32
    )

    check_model_operations(
        input_data=get_input_data(batch_size), labels=labels, input_data_corrupted=False
    )
