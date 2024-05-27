import numpy as np
import pytest
from thirdai import bolt

pytestmark = [pytest.mark.unit]


# This generates a dataset of one hot encoded vectors (with random noise added) and
# binary labels where the label is 1 if the vectors have the same index one-hot-encoded,
# and the label is 0 if the one-hot-encoded index is different.
def generate_dataset(n_classes, n_batches, batch_size, seed):
    np.random.seed(seed)
    possible_one_hot_encodings = np.eye(n_classes)

    n_samples = n_batches * batch_size

    lhs_tokens = np.random.choice(n_classes, size=n_samples).astype("uint32")
    rhs_tokens = np.random.choice(n_classes, size=n_samples).astype("uint32")
    labels_np = np.random.choice(2, size=n_samples)

    # Make the tokens the same where the label is 1
    rhs_tokens = np.where(labels_np, lhs_tokens, rhs_tokens)
    # Correct any labels in case the tokens happened to be the same by chance.
    labels_np = np.where(lhs_tokens == rhs_tokens, 1, 0)

    lhs_inputs = possible_one_hot_encodings[lhs_tokens]
    rhs_inputs = possible_one_hot_encodings[rhs_tokens]

    lhs_inputs += np.random.normal(0, 0.01, lhs_inputs.shape)
    rhs_inputs += np.random.normal(0, 0.01, rhs_inputs.shape)

    lhs_inputs = np.split(lhs_inputs, n_batches)
    rhs_inputs = np.split(rhs_inputs, n_batches)
    labels_np = np.split(labels_np, n_batches)

    data_batches = []
    label_batches = []
    for i in range(n_batches):
        data_batches.append(
            [bolt.nn.Tensor(lhs_inputs[i]), bolt.nn.Tensor(rhs_inputs[i])]
        )
        label_batches.append([bolt.nn.Tensor(labels_np[i].reshape((-1, 1)))])

    return (data_batches, label_batches), labels_np


def create_model(sim_op, input_dim, lhs_sparsity, rhs_sparsity):
    lhs_input = bolt.nn.Input(input_dim)
    rhs_input = bolt.nn.Input(input_dim)

    lhs_hidden = bolt.nn.FullyConnected(
        dim=input_dim, input_dim=input_dim, sparsity=lhs_sparsity, activation="relu"
    )(lhs_input)

    rhs_hidden = bolt.nn.FullyConnected(
        dim=input_dim, input_dim=input_dim, sparsity=rhs_sparsity, activation="relu"
    )(rhs_input)

    dot = sim_op(lhs_hidden, rhs_hidden)

    loss = bolt.nn.losses.BinaryCrossEntropy(dot, labels=bolt.nn.Input(dim=1))

    model = bolt.nn.Model(inputs=[lhs_input, rhs_input], outputs=[dot], losses=[loss])

    return model


def compute_acc(labels, scores, threshold):
    preds = np.where(scores >= threshold, 1, 0)
    return np.mean(preds == labels)


def run_similarity_test(
    sim_op, lhs_sparsity, rhs_sparsity, predict_threshold, acc_threshold
):
    n_classes = 50
    n_batches = 20
    batch_size = 100

    train_data, _ = generate_dataset(n_classes, n_batches, batch_size, seed=350924)
    test_data, test_labels_np = generate_dataset(
        n_classes, n_batches, batch_size, seed=82385
    )

    model = create_model(sim_op, n_classes, lhs_sparsity, rhs_sparsity)

    trainer = bolt.train.Trainer(model)

    trainer.train(train_data=train_data, epochs=20, learning_rate=0.01, verbose=False)

    test_outputs = []
    for inputs in test_data[0]:
        outputs = model.forward(inputs)[0]
        test_outputs.append(np.array(outputs.activations[:, 0]))

    scores = np.concatenate(test_outputs)

    labels = np.concatenate(test_labels_np)

    acc = compute_acc(labels=labels, scores=scores, threshold=predict_threshold)

    assert acc >= acc_threshold


def test_dot_product_dense_dense_embeddings():
    # Accuracy is around 0.96-0.97
    run_similarity_test(
        sim_op=bolt.nn.DotProduct(),
        lhs_sparsity=1.0,
        rhs_sparsity=1.0,
        predict_threshold=0.9,
        acc_threshold=0.8,
    )


def test_dot_product_dense_sparse_embeddings():
    # Accuracy is around 0.95-0.97
    run_similarity_test(
        sim_op=bolt.nn.DotProduct(),
        lhs_sparsity=1.0,
        rhs_sparsity=0.2,
        predict_threshold=0.98,
        acc_threshold=0.8,
    )


def test_dot_product_sparse_dense_embeddings():
    # Accuracy is around 0.95-0.97
    run_similarity_test(
        sim_op=bolt.nn.DotProduct(),
        lhs_sparsity=0.2,
        rhs_sparsity=1.0,
        predict_threshold=0.98,
        acc_threshold=0.8,
    )


def test_dot_product_sparse_sparse_embeddings():
    # Accuracy is around 0.85-0.9
    run_similarity_test(
        sim_op=bolt.nn.DotProduct(),
        lhs_sparsity=0.2,
        rhs_sparsity=0.2,
        predict_threshold=0.998,
        acc_threshold=0.7,
    )


def test_cosine_distance_dense_dense_embeddings():
    # Accuracy is around 0.98-0.99
    run_similarity_test(
        sim_op=bolt.nn.CosineSimilarity(),
        lhs_sparsity=1.0,
        rhs_sparsity=1.0,
        predict_threshold=0.6,
        acc_threshold=0.9,
    )


def test_cosine_distance_dense_sparse_embeddings():
    # Accuracy is around 0.94-0.97
    run_similarity_test(
        sim_op=bolt.nn.DotProduct(),
        lhs_sparsity=1.0,
        rhs_sparsity=0.2,
        predict_threshold=0.99,
        acc_threshold=0.8,
    )


def test_cosine_distance_sparse_dense_embeddings():
    # Accuracy is around 0.94-0.97
    run_similarity_test(
        sim_op=bolt.nn.DotProduct(),
        lhs_sparsity=0.2,
        rhs_sparsity=1.0,
        predict_threshold=0.99,
        acc_threshold=0.8,
    )


def test_cosine_distance_sparse_sparse_embeddings():
    # Accuracy is around 0.85-0.9
    run_similarity_test(
        sim_op=bolt.nn.DotProduct(),
        lhs_sparsity=0.2,
        rhs_sparsity=0.2,
        predict_threshold=0.99,
        acc_threshold=0.7,
    )
