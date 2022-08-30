import pytest

pytestmark = [pytest.mark.unit]

from thirdai import bolt, dataset
from ..utils import gen_numpy_training_data
import numpy as np


def build_single_node_bolt_dag_model(train_data, train_labels, sparsity, num_classes):
    data = dataset.from_numpy(train_data, batch_size=64)
    labels = dataset.from_numpy(train_labels, batch_size=64)

    input_layer = bolt.graph.Input(dim=num_classes)
    hidden_layer = bolt.graph.FullyConnected(
        dim=2000,
        sparsity=sparsity,
        activation="relu",
    )(input_layer)
    output_layer = bolt.graph.FullyConnected(dim=num_classes, activation="softmax")(
        hidden_layer
    )

    train_config = (
        bolt.graph.TrainConfig.make(learning_rate=0.0001, epochs=3)
        .silence()
        .with_rebuild_hash_tables(3000)
        .with_reconstruct_hash_functions(10000)
    )
    model = bolt.graph.DistributedModel(
        inputs=[input_layer],
        output=output_layer,
        train_data=[data],
        train_labels=labels,
        train_config=train_config,
        loss=bolt.CategoricalCrossEntropyLoss(),
    )
    return model


def avg_gradients(model_a, model_b):
    avg_weight_gradients_fc_1 = model_a.get_layer("fc_1").weight_gradients.copy()
    avg_weight_gradients_fc_2 = model_a.get_layer("fc_2").weight_gradients.copy()

    avg_bias_gradients_fc_1 = model_a.get_layer("fc_1").bias_gradients.copy()
    avg_bias_gradients_fc_2 = model_a.get_layer("fc_2").bias_gradients.copy()

    avg_weight_gradients_fc_1 += model_b.get_layer("fc_1").weight_gradients.copy()
    avg_weight_gradients_fc_2 += model_b.get_layer("fc_2").weight_gradients.copy()

    avg_bias_gradients_fc_1 += model_b.get_layer("fc_1").bias_gradients.copy()
    avg_bias_gradients_fc_2 += model_b.get_layer("fc_2").bias_gradients.copy()

    avg_weight_gradients_fc_1 /= 2
    avg_weight_gradients_fc_2 /= 2
    avg_bias_gradients_fc_1 /= 2
    avg_bias_gradients_fc_2 /= 2
    return (
        avg_weight_gradients_fc_1,
        avg_weight_gradients_fc_2,
        avg_bias_gradients_fc_1,
        avg_bias_gradients_fc_2,
    )


def check_models(model_a, model_b, train_x, train_y):

    FC_1_WEIGHTS = (
        model_a.get_layer("fc_1").weights.get()
        == model_b.get_layer("fc_1").weights.get()
    ).all()
    FC_2_WEIGHTS = (
        model_a.get_layer("fc_2").weights.get()
        == model_b.get_layer("fc_2").weights.get()
    ).all()
    FC_1_BIASES = (
        model_a.get_layer("fc_1").biases.get() == model_b.get_layer("fc_1").biases.get()
    ).all()
    FC_2_BIASES = (
        model_a.get_layer("fc_2").biases.get() == model_b.get_layer("fc_2").biases.get()
    ).all()

    # test_x, test_y = gen_numpy_training_data(n_samples=100)
    predict_config = (
        bolt.graph.PredictConfig.make().with_metrics(["categorical_accuracy"]).silence()
    )
    metrics_model_a = model_a.predict(
        test_data=dataset.from_numpy(train_x, batch_size=64),
        test_labels=dataset.from_numpy(train_y, batch_size=64),
        predict_config=predict_config,
    )
    metrics_model_b = model_b.predict(
        test_data=dataset.from_numpy(train_x, batch_size=64),
        test_labels=dataset.from_numpy(train_y, batch_size=64),
        predict_config=predict_config,
    )

    SAME_ACCURACY = (
        metrics_model_a[0]["categorical_accuracy"]
        == metrics_model_b[0]["categorical_accuracy"]
    )
    ACCURACY_GREATER_THAN_THRESHOLD = metrics_model_a[0]["categorical_accuracy"] > 0.8

    assert (
        FC_1_WEIGHTS and FC_2_WEIGHTS and FC_1_BIASES and FC_2_BIASES
    ), "Model Parameters are not the same across two models after training"
    assert (
        SAME_ACCURACY and ACCURACY_GREATER_THAN_THRESHOLD
    ), "Accuracy is less than threshold."


def test_distributed_training_with_bolt():
    train_x, train_y = gen_numpy_training_data(convert_to_bolt_dataset=False)

    train_x_a, train_x_b = np.split(train_x, 2)
    train_y_a, train_y_b = np.split(train_y, 2)

    model_a = build_single_node_bolt_dag_model(
        train_data=train_x_a, train_labels=train_y_a, sparsity=0.2, num_classes=10
    )
    model_b = build_single_node_bolt_dag_model(
        train_data=train_x_b, train_labels=train_y_b, sparsity=0.2, num_classes=10
    )

    num_of_batches = min(model_a.numTrainingBatch(), model_b.numTrainingBatch())

    # update same parameters across both the model
    model_b.get_layer("fc_1").weights.set(model_a.get_layer("fc_1").weights.get())
    model_b.get_layer("fc_1").biases.set(model_a.get_layer("fc_1").biases.get())
    model_b.get_layer("fc_2").weights.set(model_a.get_layer("fc_2").weights.get())
    model_b.get_layer("fc_2").biases.set(model_a.get_layer("fc_2").biases.get())

    assert (
        model_a.get_layer("fc_1").weights.get()
        == model_b.get_layer("fc_1").weights.get()
    ).all()

    epochs = 5
    for epoch in range(epochs):
        for batch_num in range(num_of_batches):
            model_a.calculateGradientSingleNode(batch_num)
            model_b.calculateGradientSingleNode(batch_num)

            (
                avg_weight_gradients_fc_1,
                avg_weight_gradients_fc_2,
                avg_bias_gradients_fc_1,
                avg_bias_gradients_fc_2,
            ) = avg_gradients(model_a=model_a, model_b=model_b)

            model_a.get_layer("fc_1").weight_gradients.set(avg_weight_gradients_fc_1)
            model_a.get_layer("fc_2").weight_gradients.set(avg_weight_gradients_fc_2)
            model_a.get_layer("fc_1").bias_gradients.set(avg_bias_gradients_fc_1)
            model_a.get_layer("fc_2").bias_gradients.set(avg_bias_gradients_fc_2)

            model_b.get_layer("fc_1").weight_gradients.set(avg_weight_gradients_fc_1)
            model_b.get_layer("fc_2").weight_gradients.set(avg_weight_gradients_fc_2)
            model_b.get_layer("fc_1").bias_gradients.set(avg_bias_gradients_fc_1)
            model_b.get_layer("fc_2").bias_gradients.set(avg_bias_gradients_fc_2)

            assert (
                model_a.get_layer("fc_1").weight_gradients.get()
                == avg_weight_gradients_fc_1
            ).all(), "Model A gradients are not equal to average Gradients"
            assert (
                model_b.get_layer("fc_1").weight_gradients.get()
                == avg_weight_gradients_fc_1
            ).all(), "Model B gradients are not equal to average Gradients"

            model_a.updateParametersSingleNode()
            model_b.updateParametersSingleNode()

    model_a.finishTraining()
    model_b.finishTraining()

    check_models(model_a=model_a, model_b=model_b, train_x=train_x, train_y=train_y)
