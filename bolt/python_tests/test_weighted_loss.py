import pytest
from thirdai import bolt, dataset
import numpy as np

from utils import gen_numpy_training_data


@pytest.mark.unit
def test_weighted_loss():
    n_classes = 20

    input_layer = bolt.nn.Input(dim=n_classes)
    hidden_layer = bolt.nn.FullyConnected(
        dim=100,
        sparsity=0.3,
        activation="relu",
        sampling_config=bolt.nn.RandomSamplingConfig(),
    )(input_layer)
    output_layer = bolt.nn.FullyConnected(dim=n_classes, activation="softmax")(
        hidden_layer
    )

    sample_weights = bolt.nn.Input(dim=1);

    model = bolt.nn.Model(inputs=[input_layer, sample_weights], output=output_layer)

    base_loss = bolt.nn.losses.CategoricalCrossEntropy()
    model.compile(bolt.nn.losses.Weighted(weights=sample_weights, loss=base_loss))

    train_x, train_y = gen_numpy_training_data(n_classes=n_classes, n_samples=2000)
    train_weights = dataset.from_numpy(np.ones(shape=2000, dtype=np.float32), batch_size=64)

    test_x, test_y = gen_numpy_training_data(n_classes=n_classes, n_samples=500)
    test_weights = dataset.from_numpy(np.ones(shape=500, dtype=np.float32), batch_size=64)

    train_cfg = bolt.TrainConfig(epochs=3, learning_rate=0.001).silence()
    model.train([train_x, train_weights], train_y, train_cfg)

    eval_cfg = bolt.EvalConfig().with_metrics(["categorical_accuracy"]).silence()
    metrics = model.evaluate([test_x, test_weights], test_y, eval_cfg)

    assert metrics[0]["categorical_accuracy"] >= 0.9
