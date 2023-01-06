# AKA Josh's Hall of Horrors

import numpy as np
import pytest
from thirdai import bolt, dataset

from utils import get_simple_concat_model

# Add a test here when you find a bug, to prevent the bug from recurring

pytestmark = [pytest.mark.unit]


def get_good_model(input_and_output_dim):
    model = get_simple_concat_model(
        hidden_layer_top_dim=10,
        hidden_layer_bottom_dim=10,
        hidden_layer_top_sparsity=0.1,
        hidden_layer_bottom_sparsity=0.1,
        num_classes=input_and_output_dim,
    )
    return model


def get_simple_train_config():
    return bolt.TrainConfig(learning_rate=0.001, epochs=3).silence()


def get_simple_eval_config():
    return bolt.EvalConfig().silence().with_metrics(["mean_squared_error"])


def get_random_dense_bolt_dataset(rows, cols):
    return dataset.from_numpy(
        np.random.rand(rows, cols).astype("float32"), batch_size=64
    )


# This test also checks that if we have an exception during training we can
# train and predict again
def test_bad_dense_input_dim():
    input_and_output_dim = 10
    num_train = 10
    model = get_good_model(input_and_output_dim)
    train_config = get_simple_train_config()
    eval_config = get_simple_eval_config()

    for bad_dim in [input_and_output_dim + dif for dif in range(-3, 3, 2)]:
        bad_dim = input_and_output_dim - 1
        data = get_random_dense_bolt_dataset(num_train, bad_dim)
        labels = get_random_dense_bolt_dataset(num_train, input_and_output_dim)
        with pytest.raises(
            ValueError,
            match=f".*Received dense BoltVector with dimension={bad_dim}, but was supposed to have dimension={input_and_output_dim}.*",
        ):
            model.train(data, labels, train_config)

    data = get_random_dense_bolt_dataset(num_train, input_and_output_dim)
    model.train(data, labels, train_config)

    for bad_dim in [input_and_output_dim + dif for dif in range(-3, 3, 2)]:
        bad_dim = input_and_output_dim - 1
        data = get_random_dense_bolt_dataset(num_train, bad_dim)
        labels = get_random_dense_bolt_dataset(num_train, input_and_output_dim)
        with pytest.raises(
            ValueError,
            match=f".*Received dense BoltVector with dimension={bad_dim}, but was supposed to have dimension={input_and_output_dim}.*",
        ):
            model.evaluate(data, labels, eval_config)

    data = get_random_dense_bolt_dataset(num_train, input_and_output_dim)
    model.evaluate(data, labels, eval_config)


def test_bad_inference_metrics():
    input_and_output_dim = 10
    num_train = 10
    model = get_good_model(input_and_output_dim)
    eval_config = get_simple_eval_config()

    data = get_random_dense_bolt_dataset(num_train, input_and_output_dim)
    labels = None
    eval_config = get_simple_eval_config().with_metrics([])
    with pytest.raises(
        ValueError,
        match=f"Doing evaluation without metrics or activations is a no-op. Did you forget to specify this in the EvalConfig?",
    ):
        model.evaluate(data, labels, eval_config)

    eval_config = get_simple_eval_config()
    with pytest.raises(
        ValueError, match=f"Cannot track accuracy metrics without labels"
    ):
        model.evaluate(data, labels, eval_config)

    eval_config = get_simple_eval_config()


# We don't check inference here too because we label handling is the same
def test_bad_label_dim_dense():
    input_and_output_dim = 10
    num_train = 10
    model = get_good_model(input_and_output_dim)
    train_config = get_simple_train_config()
    eval_config = get_simple_eval_config()

    for bad_dim in [input_and_output_dim + dif for dif in range(-3, 3, 2)]:
        bad_dim = input_and_output_dim - 1
        labels = get_random_dense_bolt_dataset(num_train, bad_dim)
        data = get_random_dense_bolt_dataset(num_train, input_and_output_dim)
        with pytest.raises(
            ValueError,
            match=f".*Received dense BoltVector with dimension={bad_dim}, but was supposed to have dimension={input_and_output_dim}.*",
        ):
            model.train(data, labels, train_config)


def test_label_train_num_example_mismatch():
    input_and_output_dim = 10
    num_examples = 11
    num_labels = 10
    model = get_good_model(input_and_output_dim)
    train_config = get_simple_train_config()
    eval_config = get_simple_eval_config()
    data = get_random_dense_bolt_dataset(num_examples, input_and_output_dim)
    labels = get_random_dense_bolt_dataset(num_labels, input_and_output_dim)

    with pytest.raises(
        ValueError,
        match=f".*found {num_labels} samples in one dataset and {num_examples} samples in another.*",
    ):
        model.train(data, labels, train_config)

    with pytest.raises(
        ValueError,
        match=f".*found {num_labels} samples in one dataset and {num_examples} samples in another.*",
    ):
        model.evaluate(data, labels, eval_config)
