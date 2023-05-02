import pytest
from thirdai import bolt, hashing

from utils import gen_numpy_training_data

N_CLASSES = 200


def build_model():
    input_layer = bolt.nn.Input(dim=N_CLASSES)

    hidden_layer = bolt.nn.FullyConnected(dim=100, activation="relu")(input_layer)

    output_layer = bolt.nn.FullyConnected(
        dim=N_CLASSES, sparsity=0.2, activation="softmax"
    )(hidden_layer)

    model = bolt.nn.Model(inputs=[input_layer], output=output_layer)
    model.compile(bolt.nn.losses.CategoricalCrossEntropy())

    return model


@pytest.mark.unit
def test_get_set_hash_tables():
    # This test checks that copying only the weights doesn't preserve accuracy
    # when using sparse inference.
    model = build_model()

    train_data, train_labels = gen_numpy_training_data(
        n_classes=N_CLASSES, n_samples=10000
    )
    test_data, test_labels = gen_numpy_training_data(
        n_classes=N_CLASSES, n_samples=1000
    )

    train_cfg = (
        bolt.TrainConfig(epochs=5, learning_rate=0.001)
        .with_rebuild_hash_tables(4)
        .with_reconstruct_hash_functions(20)
    )
    model.train(train_data, train_labels, train_cfg)

    eval_cfg = (
        bolt.EvalConfig()
        .with_metrics(["categorical_accuracy"])
        .enable_sparse_inference()
    )

    (metrics,) = model.evaluate(test_data, test_labels, eval_cfg)

    assert metrics["categorical_accuracy"] >= 0.75

    new_model = build_model()

    new_model.get_layer("fc_1").weights.set(model.get_layer("fc_1").weights.get())
    new_model.get_layer("fc_1").biases.set(model.get_layer("fc_1").biases.get())
    new_model.get_layer("fc_2").weights.set(model.get_layer("fc_2").weights.get())
    new_model.get_layer("fc_2").biases.set(model.get_layer("fc_2").biases.get())

    (new_metrics,) = new_model.evaluate(test_data, test_labels, eval_cfg)

    # Check that the accuracy is bad before copying the hash tables too
    assert new_metrics["categorical_accuracy"] <= 0.4

    hash_fn_save_file = "./temp_saved_hash_fn"
    hash_table_save_file = "./temp_saved_hash_table"

    hash_fn, hash_table = model.get_layer("fc_2").get_hash_table()

    hash_fn.save(hash_fn_save_file)
    hash_fn = hashing.DWTA.load(hash_fn_save_file)

    hash_table.save(hash_table_save_file)
    hash_table = bolt.nn.HashTable.load(hash_table_save_file)

    new_model.get_layer("fc_2").set_hash_table(hash_fn, hash_table)

    (new_metrics,) = new_model.evaluate(test_data, test_labels, eval_cfg)

    assert new_metrics["categorical_accuracy"] >= 0.75
