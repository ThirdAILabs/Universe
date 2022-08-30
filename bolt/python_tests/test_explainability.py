from thirdai import bolt, dataset
import pytest
import numpy as np
from utils import gen_numpy_training_data

pytestmark = [pytest.mark.unit]


def test_get_input_gradients():
    n_classes = 100
    batch_size = 100
    input_layer = bolt.graph.Input(dim=n_classes)
    hidden_layer = bolt.graph.FullyConnected(dim=200, activation="relu")(input_layer)
    output_layer = bolt.graph.FullyConnected(dim=n_classes, activation="softmax")(
        hidden_layer
    )

    model = bolt.graph.Model(inputs=[input_layer], output=output_layer)
    model.compile(bolt.CategoricalCrossEntropyLoss())

    train_x, train_y = gen_numpy_training_data(n_classes=n_classes, n_samples=2000)

    test_x_np, test_y_np = gen_numpy_training_data(
        n_classes=n_classes, convert_to_bolt_dataset=False
    )
    test_x = dataset.from_numpy(test_x_np, batch_size=batch_size)

    train_cfg = bolt.graph.TrainConfig.make(learning_rate=0.001, epochs=10)

    model.train(
        train_x,
        train_y,
        train_cfg,
    )

    correct_explainations = 0
    total = 0

    for batch_idx in range(len(test_x)):
        for vec_idx in range(len(test_x[batch_idx])):
            explain = model.explain_prediction([test_x[batch_idx][vec_idx]])
            explain_np = np.array(explain)

            if np.argmax(explain_np) == test_y_np[batch_idx * batch_size + vec_idx]:
                correct_explainations += 1
            total += 1

    assert correct_explainations >= 0.8 * total
