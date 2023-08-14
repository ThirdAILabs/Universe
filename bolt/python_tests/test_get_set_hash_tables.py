import pytest
import thirdai
from thirdai import bolt, hashing

from utils import gen_numpy_training_data

N_CLASSES = 200


def build_model():
    input_layer = bolt.nn.Input(dim=N_CLASSES)

    hidden_layer = bolt.nn.FullyConnected(
        dim=100, input_dim=input_layer.dim(), activation="relu"
    )(input_layer)

    output_layer = bolt.nn.FullyConnected(
        dim=N_CLASSES,
        input_dim=hidden_layer.dim(),
        sparsity=0.2,
        activation="softmax",
    )(hidden_layer)

    loss = bolt.nn.losses.CategoricalCrossEntropy(
        output_layer, labels=bolt.nn.Input(dim=N_CLASSES)
    )

    model = bolt.nn.Model(inputs=[input_layer], outputs=[output_layer], losses=[loss])

    return model


def get_data(n_samples):
    x, y = gen_numpy_training_data(n_classes=N_CLASSES, n_samples=n_samples)

    x = bolt.train.convert_dataset(x, dim=N_CLASSES)
    y = bolt.train.convert_dataset(y, dim=N_CLASSES)

    return x, y


@pytest.mark.unit
def test_get_set_hash_tables():
    # This test checks that copying only the weights doesn't preserve accuracy
    # when using sparse inference.
    model = build_model()

    train_data = gen_numpy_training_data(n_classes=N_CLASSES, n_samples=10000)
    test_data = gen_numpy_training_data(n_classes=N_CLASSES, n_samples=1000)

    trainer = bolt.train.Trainer(model)

    trainer.train(
        train_data,
        epochs=5,
        learning_rate=0.001,
    )

    metrics = trainer.validate(
        test_data, validation_metrics=["categorical_accuracy"], use_sparsity=True
    )

    assert metrics["val_categorical_accuracy"][-1] >= 0.70

    new_model = build_model()
    new_trainer = bolt.train.Trainer(new_model)

    for old_op, new_op in zip(model.ops(), new_model.ops()):
        new_op.set_weights(old_op.weights)
        new_op.set_biases(old_op.biases)

    new_metrics = new_trainer.validate(
        test_data, validation_metrics=["categorical_accuracy"], use_sparsity=True
    )

    hash_fn_save_file = "./temp_saved_hash_fn"
    hash_table_save_file = "./temp_saved_hash_table"

    hash_fn, hash_table = model.ops()[1].get_hash_table()

    hash_fn.save(hash_fn_save_file)
    hash_fn = hashing.DWTA.load(hash_fn_save_file)

    hash_table.save(hash_table_save_file)
    hash_table = thirdai.bolt.nn.HashTable.load(hash_table_save_file)

    new_model.ops()[1].set_hash_table(hash_fn, hash_table)

    new_metrics = new_trainer.validate(
        test_data, validation_metrics=["categorical_accuracy"], use_sparsity=True
    )

    assert new_metrics["val_categorical_accuracy"][-1] >= 0.70
