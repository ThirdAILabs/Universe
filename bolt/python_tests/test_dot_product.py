import pytest
from thirdai import bolt, dataset
import numpy as np
from sklearn.metrics import roc_auc_score

pytestmark = [pytest.mark.unit]


def generate_dataset(n_classes, n_samples, batch_size):
    possible_one_hot_encodings = np.eye(n_classes)

    lhs_tokens = np.random.choice(n_classes, size=n_samples).astype("uint32")
    rhs_tokens = np.random.choice(n_classes, size=n_samples).astype("uint32")
    labels_np = np.random.choice(2, size=n_samples)

    # Make the tokens the same where the label is 1
    rhs_tokens = np.where(labels_np, lhs_tokens, rhs_tokens)

    lhs_inputs = possible_one_hot_encodings[lhs_tokens]
    rhs_inputs = possible_one_hot_encodings[rhs_tokens]

    lhs_inputs += np.random.normal(0, 0.1, lhs_inputs.shape)
    rhs_inputs += np.random.normal(0, 0.1, rhs_inputs.shape)

    lhs_dataset = dataset.from_numpy(lhs_inputs.astype("float32"), batch_size)
    rhs_dataset = dataset.from_numpy(rhs_inputs.astype("float32"), batch_size)

    labels_dataset = dataset.from_numpy(labels_np.astype("float32"), batch_size)

    return lhs_dataset, rhs_dataset, labels_dataset, labels_np


def create_model(input_dim, lhs_sparsity, rhs_sparsity):
    lhs_input = bolt.graph.Input(input_dim)
    rhs_input = bolt.graph.Input(input_dim)

    lhs_hidden = bolt.graph.FullyConnected(
        dim=200, sparsity=lhs_sparsity, activation="relu"
    )(lhs_input)

    rhs_hidden = bolt.graph.FullyConnected(
        dim=200, sparsity=rhs_sparsity, activation="relu"
    )(rhs_input)

    dot = bolt.graph.DotProduct()(lhs_hidden, rhs_hidden)

    model = bolt.graph.Model(inputs=[lhs_input, rhs_input], output=dot)

    model.compile(bolt.BinaryCrossEntropyLoss())

    return model


def run_dot_product_test(lhs_sparsity, rhs_sparsity):
    n_classes = 100
    n_samples = 2000
    batch_size = 100

    train_rhs_data, train_lhs_data, train_labels, _ = generate_dataset(
        n_classes, n_samples, batch_size
    )
    test_rhs_data, test_lhs_data, test_labels, test_labels_np = generate_dataset(
        n_classes, n_samples, batch_size
    )

    model = create_model(n_classes, lhs_sparsity, rhs_sparsity)

    train_cfg = bolt.graph.TrainConfig.make(learning_rate=0.01, epochs=20).silence()
    predict_cfg = bolt.graph.PredictConfig.make().return_activations().silence()

    model.train([train_lhs_data, train_rhs_data], train_labels, train_cfg)
    _, activations = model.predict(
        [test_lhs_data, test_rhs_data], test_labels, predict_cfg
    )

    roc_auc = roc_auc_score(test_labels_np, activations[:, 0])

    assert roc_auc >= 0.8


def test_dot_product_dense_dense_embeddings():
    run_dot_product_test(lhs_sparsity=1.0, rhs_sparsity=1.0)


def test_dot_product_dense_sparse_embeddings():
    run_dot_product_test(lhs_sparsity=1.0, rhs_sparsity=0.2)


def test_dot_product_sparse_dense_embeddings():
    run_dot_product_test(lhs_sparsity=0.2, rhs_sparsity=1.0)


def test_dot_product_sparse_sparse_embeddings():
    run_dot_product_test(lhs_sparsity=0.2, rhs_sparsity=0.2)
