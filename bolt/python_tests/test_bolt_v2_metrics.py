import numpy as np
import pytest
from thirdai import bolt_v2 as bolt
from thirdai import dataset

N_CLASSES = 10


def build_model(metric_to_test, **kwargs):
    # This operator is useless, but we need to create a 1 layer model to test metrics
    op = bolt.nn.FullyConnected(
        dim=N_CLASSES,
        input_dim=N_CLASSES,
        sparsity=0.4,
        activation="relu",
    )

    input_layer = bolt.nn.Input(dim=N_CLASSES)

    output_layer = op(input_layer)

    labels = bolt.nn.Input(dim=N_CLASSES)

    loss = bolt.nn.losses.CategoricalCrossEntropy(
        activations=output_layer, labels=labels
    )

    model = bolt.nn.Model(inputs=[input_layer], outputs=[output_layer], losses=[loss])

    # We don't care about model output, we simply care about the correctness of the metric function
    metrics = {
        k: metric_to_test[k](outputs=input_layer, labels=labels, **kwargs)
        for k in metric_to_test
    }

    return model, metrics


def lists_to_data(x, y):
    x = bolt.train.convert_dataset(
        dataset.from_numpy(np.array(x).astype("float32"), batch_size=1), dim=N_CLASSES
    )
    y = bolt.train.convert_dataset(
        dataset.from_numpy(np.array(y).astype("float32"), batch_size=1), dim=N_CLASSES
    )
    return (x, y)


def get_metric(model, test_data, metrics, metric_name):
    trainer = bolt.train.Trainer(model)
    metric_vals = trainer.validate(
        validation_data=test_data, validation_metrics=metrics
    )
    return metric_vals[metric_name][-1]


@pytest.mark.unit
def test_precision_at_1():
    metric_name = "prec@1"
    k = 1
    model, metrics = build_model({metric_name: bolt.train.metrics.PrecisionAtK}, k=k)

    test_data = lists_to_data(
        x=[[0.1, 0.1, 0.1, 1.0, 5.0, 0.0, 0.0, 0.1, 0.1, 6.0]],
        y=[[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
    )
    metric_val = get_metric(model, test_data, metrics, metric_name)
    assert np.allclose(metric_val, 1)

    test_data = lists_to_data(
        x=[[0.1, 0.1, 1.1, 1.0, 5.0, 2.0, 3.0, 0.1, 0.1, 6.0]],
        y=[[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0]],
    )
    metric_val = get_metric(model, test_data, metrics, metric_name)
    assert np.allclose(metric_val, 1)

    test_data = lists_to_data(
        x=[[0.1, 0.1, 0.1, 1.0, 5.0, 0.0, 0.0, 0.1, 0.1, 6.0]],
        y=[[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]],
    )
    metric_val = get_metric(model, test_data, metrics, metric_name)
    assert np.allclose(metric_val, 0)

    # Following two tests show tie behavior, largest index is chosen if values are tied
    test_data = lists_to_data(
        x=[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
        y=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
    )
    metric_val = get_metric(model, test_data, metrics, metric_name)
    assert np.allclose(metric_val, 1)

    test_data = lists_to_data(
        x=[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
        y=[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    )
    metric_val = get_metric(model, test_data, metrics, metric_name)
    assert np.allclose(metric_val, 0)


@pytest.mark.unit
def test_precision_at_5():
    metric_name = "prec@5"
    k = 5
    model, metrics = build_model({metric_name: bolt.train.metrics.PrecisionAtK}, k=k)

    test_data = lists_to_data(
        x=[[0.1, 0.1, 0.1, 1.0, 5.0, 0.0, 0.0, 0.1, 0.1, 6.0]],
        y=[[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
    )
    metric_val = get_metric(model, test_data, metrics, metric_name)
    assert np.allclose(metric_val, 0.4)

    test_data = lists_to_data(
        x=[[0.1, 0.1, 1.1, 1.0, 5.0, 2.0, 3.0, 0.1, 0.1, 6.0]],
        y=[[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0]],
    )
    metric_val = get_metric(model, test_data, metrics, metric_name)
    assert np.allclose(metric_val, 1)

    test_data = lists_to_data(
        x=[[0.1, 0.1, 0.1, 1.0, 5.0, 0.0, 0.0, 0.1, 0.1, 6.0]],
        y=[[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]],
    )
    metric_val = get_metric(model, test_data, metrics, metric_name)
    assert np.allclose(metric_val, 0)


@pytest.mark.unit
def test_precision_at_10():
    metric_name = "prec@10"
    k = 10
    model, metrics = build_model({metric_name: bolt.train.metrics.PrecisionAtK}, k=k)

    test_data = lists_to_data(
        x=[[0.1, 0.1, 0.1, 1.0, 5.0, 0.0, 0.0, 0.1, 0.1, 6.0]],
        y=[[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
    )
    metric_val = get_metric(model, test_data, metrics, metric_name)
    assert np.allclose(metric_val, 0.2)

    test_data = lists_to_data(
        x=[[0.1, 0.1, 1.1, 1.0, 5.0, 2.0, 3.0, 0.1, 0.1, 6.0]],
        y=[[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0]],
    )
    metric_val = get_metric(model, test_data, metrics, metric_name)
    assert np.allclose(metric_val, 0.5)

    test_data = lists_to_data(
        x=[[0.1, 0.1, 0.1, 1.0, 5.0, 0.0, 0.0, 0.1, 0.1, 6.0]],
        y=[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]],
    )
    metric_val = get_metric(model, test_data, metrics, metric_name)
    assert np.allclose(metric_val, 0.9)


@pytest.mark.unit
def test_recall_at_1():
    metric_name = "rec@1"
    k = 1
    model, metrics = build_model({metric_name: bolt.train.metrics.RecallAtK}, k=k)

    test_data = lists_to_data(
        x=[[0.1, 0.1, 0.1, 1.0, 5.0, 0.0, 0.0, 0.1, 0.1, 6.0]],
        y=[[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
    )
    metric_val = get_metric(model, test_data, metrics, metric_name)
    assert np.allclose(metric_val, 0.5)

    test_data = lists_to_data(
        x=[[0.1, 0.1, 1.1, 1.0, 5.0, 2.0, 3.0, 0.1, 0.1, 6.0]],
        y=[[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0]],
    )
    metric_val = get_metric(model, test_data, metrics, metric_name)
    assert np.allclose(metric_val, 0.2)

    test_data = lists_to_data(
        x=[[0.1, 0.1, 0.1, 1.0, 5.0, 7.0, 0.0, 0.1, 0.1, 6.0]],
        y=[[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]],
    )
    metric_val = get_metric(model, test_data, metrics, metric_name)
    assert np.allclose(metric_val, 1)


@pytest.mark.unit
def test_recall_at_5():
    metric_name = "rec@5"
    k = 5

    test_data = lists_to_data(
        x=[[0.1, 0.1, 0.1, 1.0, 5.0, 0.0, 0.0, 0.1, 0.1, 6.0]],
        y=[[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
    )
    model, metrics = build_model({metric_name: bolt.train.metrics.RecallAtK}, k=k)
    metric_val = get_metric(model, test_data, metrics, metric_name)
    assert np.allclose(metric_val, 1)

    test_data = lists_to_data(
        x=[[0.1, 0.1, 1.1, 1.0, 5.0, 2.0, 3.0, 0.1, 0.1, 6.0]],
        y=[[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0]],
    )
    model, metrics = build_model({metric_name: bolt.train.metrics.RecallAtK}, k=k)
    metric_val = get_metric(model, test_data, metrics, metric_name)
    assert np.allclose(metric_val, 1)

    test_data = lists_to_data(
        x=[[0.1, 0.1, 0.1, 1.0, 5.0, 0.0, 0.0, 0.1, 0.1, 6.0]],
        y=[[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]],
    )
    model, metrics = build_model({metric_name: bolt.train.metrics.RecallAtK}, k=k)
    metric_val = get_metric(model, test_data, metrics, metric_name)
    assert np.allclose(metric_val, 0)


@pytest.mark.unit
def test_recall_at_10():
    metric_name = "rec@10"
    k = 10

    test_data = lists_to_data(
        x=[[0.1, 0.1, 0.1, 1.0, 5.0, 0.0, 0.0, 0.1, 0.1, 6.0]],
        y=[[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
    )
    model, metrics = build_model({metric_name: bolt.train.metrics.RecallAtK}, k=k)
    metric_val = get_metric(model, test_data, metrics, metric_name)
    assert np.allclose(metric_val, 1)

    test_data = lists_to_data(
        x=[[0.1, 0.1, 1.1, 1.0, 5.0, 2.0, 3.0, 0.1, 0.1, 6.0]],
        y=[[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0]],
    )
    model, metrics = build_model({metric_name: bolt.train.metrics.RecallAtK}, k=k)
    metric_val = get_metric(model, test_data, metrics, metric_name)
    assert np.allclose(metric_val, 1)

    test_data = lists_to_data(
        x=[[0.1, 0.1, 0.1, 1.0, 5.0, 0.0, 0.0, 0.1, 0.1, 6.0]],
        y=[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]],
    )
    model, metrics = build_model({metric_name: bolt.train.metrics.RecallAtK}, k=k)
    metric_val = get_metric(model, test_data, metrics, metric_name)
    assert np.allclose(metric_val, 1)
