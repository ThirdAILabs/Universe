from thirdai import bolt, dataset
import pytest
import numpy as np
from utils import gen_numpy_training_data, get_simple_dag_model

pytestmark = [pytest.mark.unit]
n_classes = 100
batch_size = 100


def get_trained_model(n_samples):
    model = get_simple_dag_model(
        input_dim=n_classes,
        hidden_layer_dim=200,
        hidden_layer_sparsity=1,
        output_dim=n_classes,
    )

    train_x, train_y = gen_numpy_training_data(n_classes=n_classes, n_samples=n_samples)

    train_cfg = bolt.graph.TrainConfig.make(learning_rate=0.001, epochs=10)

    model.train(
        train_x,
        train_y,
        train_cfg,
    )

    return model


def test_get_input_gradients():
    model = get_trained_model(n_samples=2000)

    test_x_np, test_y_np = gen_numpy_training_data(
        n_classes=n_classes, convert_to_bolt_dataset=False
    )
    test_x = dataset.from_numpy(test_x_np, batch_size=batch_size)

    correct_explainations = 0
    total = 0

    for batch_idx in range(len(test_x)):
        for vec_idx in range(len(test_x[batch_idx])):
            explain = model.explain_prediction([test_x[batch_idx][vec_idx]])
            explain = explain * test_x_np[batch_idx * batch_size + vec_idx]
            explain_np = np.array(explain)

            if np.argmax(explain_np) == test_y_np[batch_idx * batch_size + vec_idx]:
                correct_explainations += 1
            total += 1

    assert correct_explainations >= 0.8 * total


def test_get_input_gradients_load_and_save():
    model = get_trained_model(n_samples=200)

    test_x_np, _ = gen_numpy_training_data(
        n_classes=n_classes, convert_to_bolt_dataset=False
    )
    test_x = dataset.from_numpy(test_x_np, batch_size=batch_size)

    model_saved_file = "saved_model"

    model.save(model_saved_file)

    loaded_model = bolt.graph.Model.load(model_saved_file)

    for batch_idx in range(len(test_x)):
        for vec_idx in range(len(test_x[batch_idx])):
            explain = model.explain_prediction([test_x[batch_idx][vec_idx]])
            load_explain = loaded_model.explain_prediction([test_x[batch_idx][vec_idx]])
            assert explain == load_explain
