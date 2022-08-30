from ..utils import gen_numpy_training_data
import pytest
from thirdai import bolt


@pytest.mark.unit
def test_dropout_layer():
    n_classes = 20

    input_layer = bolt.graph.Input(dim=n_classes)
    hidden_layer = bolt.graph.FullyConnected(
        dim=100,
        sparsity=0.3,
        activation="relu",
        sampling_config=bolt.RandomSamplingConfig(),
    )(input_layer)
    output_layer = bolt.graph.FullyConnected(dim=n_classes, activation="softmax")(
        hidden_layer
    )

    model = bolt.graph.Model(inputs=[input_layer], output=output_layer)

    model.compile(bolt.CategoricalCrossEntropyLoss())

    train_x, train_y = gen_numpy_training_data(n_classes=n_classes, n_samples=2000)
    test_x, test_y = gen_numpy_training_data(n_classes=n_classes, n_samples=500)

    train_cfg = bolt.graph.TrainConfig.make(epochs=3, learning_rate=0.001)
    model.train(train_x, train_y, train_cfg)

    predict_cfg = bolt.graph.PredictConfig.make().with_metrics(["categorical_accuracy"])
    metrics = model.predict(test_x, test_y, predict_cfg)

    assert metrics[0]["categorical_accuracy"] >= 0.9
