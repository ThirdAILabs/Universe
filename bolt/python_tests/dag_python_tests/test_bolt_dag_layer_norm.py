import pytest
from thirdai import bolt
from ..utils import gen_numpy_training_data

BATCH_SIZE = 32
LEARNING_RATE = 0.001
ACCURACY_THRESHOLD = 0.95
EPOCHS = 5


def get_simple_model(num_classes, sparsity=1.0):
    input_layer = bolt.graph.Input(dim=num_classes)
    hidden_layer = bolt.graph.FullyConnected(
        dim=num_classes, activation="relu", sparsity=sparsity
    )(input_layer)

    # By default normalization applies scaling and centering
    layer_norm_config = (
        bolt.graph.LayerNormConfig.make()
        .center(beta_regularizer=0.0025)
        .scale(gamma_regularizer=0.9)
    )

    normalization_layer = bolt.graph.LayerNormalization(
        layer_norm_config=layer_norm_config
    )(hidden_layer)

    output_layer = bolt.graph.FullyConnected(dim=100, activation="softmax")(
        normalization_layer
    )

    model = bolt.graph.Model(inputs=[input_layer], output=output_layer)
    model.compile(loss=bolt.CategoricalCrossEntropyLoss())

    return model


@pytest.mark.unit
def test_normalize_layer_activations():

    model_with_normalization = get_simple_model(num_classes=100)

    train_data, train_labels = gen_numpy_training_data(
        n_classes=100, n_samples=10000, batch_size_for_conversion=BATCH_SIZE
    )

    train_config = bolt.graph.TrainConfig.make(
        learning_rate=LEARNING_RATE, epochs=EPOCHS
    ).silence()

    model_with_normalization.train(
        train_data=train_data, train_labels=train_labels, train_config=train_config
    )

    predict_config = bolt.graph.PredictConfig.make().with_metrics(
        ["categorical_accuracy"]
    )

    metrics = model_with_normalization.predict(
        test_data=train_data, test_labels=train_labels, predict_config=predict_config
    )

    print("categorical_acc = {}\n".format(metrics[0]["categorical_accuracy"]))
    assert metrics[0]["categorical_accuracy"] >= ACCURACY_THRESHOLD
