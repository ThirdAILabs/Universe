from thirdai import bolt_v2 as bolt
import pytest
from utils import gen_numpy_training_data
from thirdai import dataset
import numpy as np


def build_model(n_classes):
    vector_input = bolt.nn.Input(dim=n_classes)

    hidden = bolt.nn.FullyConnected(dim=150, input_dim=n_classes, activation="relu")(
        vector_input
    )

    token_input = bolt.nn.Input(dim=n_classes)

    embedding = bolt.nn.Embedding(
        num_embedding_lookups=8,
        lookup_size=8,
        log_embedding_block_size=10,
        update_chunk_size=8,
        reduction="sum",
    )(token_input)

    concat = bolt.nn.Concatenate()([hidden, embedding])

    output1 = bolt.nn.FullyConnected(
        dim=n_classes, input_dim=150, activation="softmax"
    )(hidden)

    output2 = bolt.nn.FullyConnected(
        dim=n_classes, input_dim=214, activation="sigmoid"
    )(concat)

    output3 = bolt.nn.FullyConnected(dim=n_classes, input_dim=64, activation="softmax")(
        embedding
    )

    labels = bolt.nn.Input(dim=n_classes)

    loss1 = bolt.nn.losses.CategoricalCrossEntropy(activations=output1, labels=labels)
    loss2 = bolt.nn.losses.BinaryCrossEntropy(activations=output2, labels=labels)
    loss3 = bolt.nn.losses.CategoricalCrossEntropy(activations=output3, labels=labels)

    model = bolt.nn.Model(
        inputs=[vector_input, token_input],
        outputs=[output1, output2, output3],
        losses=[loss1, loss2, loss3],
    )

    return model


@pytest.mark.unit
def test_bolt_save_load():
    N_CLASSES = 100
    model = build_model(N_CLASSES)

    train_data, train_labels = gen_numpy_training_data(
        n_classes=N_CLASSES, n_samples=2000
    )

    train_data = bolt.train.convert_dataset(train_data, dim=N_CLASSES)
    train_labels = bolt.train.convert_dataset(train_labels, dim=N_CLASSES)

    def train_model():
        for x, y in zip(train_data, train_labels):
            model.train_on_batch([x, y], [y, y, y])
            model.update_parameters(learning_rate=0.1)

    test_data, test_labels_np = gen_numpy_training_data(
        n_classes=N_CLASSES, n_samples=1000, convert_to_bolt_dataset=False
    )

    test_data = dataset.from_numpy(test_data, len(test_data))
    test_labels = dataset.from_numpy(test_labels_np, len(test_labels_np))
    test_data = bolt.train.convert_dataset(test_data, dim=N_CLASSES)
    test_labels = bolt.train.convert_dataset(test_labels, dim=N_CLASSES)

    def test_model():
        outputs = model.forward([test_data[0], test_labels[0]], use_sparsity=False)
        for output in outputs:
            predictions = np.argmax(output.activations, axis=1)
            acc = np.mean(predictions == test_labels_np)

            assert acc >= 0.8

    train_model()

    test_model()

    temp_save_path = "./temp_save_model"
    model.save(temp_save_path)

    model = bolt.nn.Model.load(temp_save_path)

    test_model()

    train_model()

    test_model()
