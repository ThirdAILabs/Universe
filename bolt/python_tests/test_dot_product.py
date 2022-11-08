import numpy as np
import pytest
from thirdai import bolt, dataset

pytestmark = [pytest.mark.unit]


# This generates a dataset of one hot encoded vectors (with random noise added) and 
# binary labels where the label is 1 if the vectors have the same index one-hot-encoded, 
# and the label is 0 if the one-hot-encoded index is different.
def generate_dataset(n_classes, n_samples, batch_size):
    possible_one_hot_encodings = np.eye(n_classes)

    lhs_tokens = np.random.choice(n_classes, size=n_samples).astype("uint32")
    rhs_tokens = np.random.choice(n_classes, size=n_samples).astype("uint32")
    labels_np = np.random.choice(2, size=n_samples)

    # Make the tokens the same where the label is 1
    rhs_tokens = np.where(labels_np, lhs_tokens, rhs_tokens)
    # Correct any labels in case the tokens happened to be the same by chance.
    labels_np = np.where(lhs_tokens == rhs_tokens, 1, 0)

    lhs_inputs = possible_one_hot_encodings[lhs_tokens]
    rhs_inputs = possible_one_hot_encodings[rhs_tokens]

    lhs_inputs += np.random.normal(0, 0.1, lhs_inputs.shape)
    rhs_inputs += np.random.normal(0, 0.1, rhs_inputs.shape)

    lhs_dataset = dataset.from_numpy(lhs_inputs.astype("float32"), batch_size)
    rhs_dataset = dataset.from_numpy(rhs_inputs.astype("float32"), batch_size)

    labels_dataset = dataset.from_numpy(labels_np.astype("float32"), batch_size)

    return lhs_dataset, rhs_dataset, labels_dataset, labels_np


def create_model(input_dim, lhs_sparsity, rhs_sparsity):
    lhs_input = bolt.nn.Input(input_dim)
    rhs_input = bolt.nn.Input(input_dim)

    lhs_hidden = bolt.nn.FullyConnected(
        dim=200, sparsity=lhs_sparsity, activation="relu"
    )(lhs_input)

    rhs_hidden = bolt.nn.FullyConnected(
        dim=200, sparsity=rhs_sparsity, activation="relu"
    )(rhs_input)

    dot = bolt.nn.DotProduct()(lhs_hidden, rhs_hidden)

    model = bolt.nn.Model(inputs=[lhs_input, rhs_input], output=dot)

    model.compile(bolt.nn.losses.BinaryCrossEntropy())

    return model


def compute_acc(labels, scores, threshold):
    preds = np.where(scores >= threshold, 1, 0)
    return np.mean(preds == labels)


def run_dot_product_test(lhs_sparsity, rhs_sparsity, predict_threshold, acc_threshold):
    n_classes = 50
    n_samples = 2000
    batch_size = 100

    train_rhs_data, train_lhs_data, train_labels, _ = generate_dataset(
        n_classes, n_samples, batch_size
    )
    test_rhs_data, test_lhs_data, test_labels, test_labels_np = generate_dataset(
        n_classes, n_samples, batch_size
    )

    model = create_model(n_classes, lhs_sparsity, rhs_sparsity)

    train_cfg = bolt.TrainConfig(learning_rate=0.01, epochs=20).silence()
    eval_cfg = bolt.EvalConfig().return_activations().silence()

    model.train([train_lhs_data, train_rhs_data], train_labels, train_cfg)
    _, activations = model.evaluate(
        [test_lhs_data, test_rhs_data], test_labels, eval_cfg
    )

    scores = activations[:, 0]
    acc = compute_acc(labels=test_labels_np, scores=scores, threshold=predict_threshold)

    assert acc >= acc_threshold


def test_dot_product_dense_dense_embeddings():
    # Accuracy is around 0.96-0.97
    run_dot_product_test(
        lhs_sparsity=1.0, rhs_sparsity=1.0, predict_threshold=0.9, acc_threshold=0.8
    )


def test_dot_product_dense_sparse_embeddings():
    # Accuracy is around 0.95-0.97
    run_dot_product_test(
        lhs_sparsity=1.0, rhs_sparsity=0.2, predict_threshold=0.98, acc_threshold=0.8
    )


def test_dot_product_sparse_dense_embeddings():
    # Accuracy is around 0.95-0.97
    run_dot_product_test(
        lhs_sparsity=0.2, rhs_sparsity=1.0, predict_threshold=0.98, acc_threshold=0.8
    )


def test_dot_product_sparse_sparse_embeddings():
    # Accuracy is around 0.85-0.9
    run_dot_product_test(
        lhs_sparsity=0.2, rhs_sparsity=0.2, predict_threshold=0.998, acc_threshold=0.7
    )
