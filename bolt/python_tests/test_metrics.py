import numpy as np
import pytest
from thirdai import bolt, dataset

LABEL_DIM = 5

pytestmark = [pytest.mark.unit]


def build_metrics_test_model(metric, metric_name, metric_args={}):
    # This operator is not used, but we need to create a 1 layer model to test metrics
    op = bolt.nn.FullyConnected(
        dim=LABEL_DIM,
        input_dim=LABEL_DIM,
        sparsity=0.4,
        activation="relu",
    )

    input_layer = bolt.nn.Input(dim=LABEL_DIM)

    output_layer = op(input_layer)

    labels = bolt.nn.Input(dim=LABEL_DIM)

    loss = bolt.nn.losses.CategoricalCrossEntropy(
        activations=output_layer, labels=labels
    )

    model = bolt.nn.Model(inputs=[input_layer], outputs=[output_layer], losses=[loss])

    # We don't care about model output, we simply care about the correctness of the metric
    # function, so we apply the metric to the input as a workaround
    metrics = {metric_name: metric(outputs=input_layer, labels=labels, **metric_args)}

    return model, metrics


def lists_to_data(x, y):
    x = bolt.train.convert_dataset(
        dataset.from_numpy(np.array(x).astype("float32"), batch_size=1), dim=LABEL_DIM
    )
    y = bolt.train.convert_dataset(
        dataset.from_numpy(np.array(y).astype("float32"), batch_size=1), dim=LABEL_DIM
    )
    return x, y


def get_metric(model, test_data, metrics, metric_name):
    trainer = bolt.train.Trainer(model)
    metric_vals = trainer.validate(
        validation_data=test_data, validation_metrics=metrics
    )
    return metric_vals[metric_name][-1]


def evaluate_test_cases(test_cases, model, metrics, metric_name):
    for tc in test_cases:
        test_data = lists_to_data(x=tc["x"], y=tc["y"])
        metric_val = get_metric(model, test_data, metrics, metric_name)
        assert np.isclose(metric_val, tc["correct_metric_val"])


def test_precision_at_1():
    metric_name = "prec@1"
    k = 1
    model, metrics = build_metrics_test_model(
        bolt.train.metrics.PrecisionAtK, metric_name, {"k": k}
    )

    test_cases = [
        {
            "x": [[0.1, 0.1, 0.1, 1.0, 5.0]],
            "y": [[0.0, 0.0, 0.0, 0.0, 1.0]],
            "correct_metric_val": 1,
        },
        {
            "x": [[0.1, 0.1, 5.0, 1.0, 3.0]],
            "y": [[0.0, 0.0, 1.0, 0.0, 1.0]],
            "correct_metric_val": 1,
        },
        {
            "x": [[0.1, 0.1, 0.1, 1.0, 5.0]],
            "y": [[0.0, 0.0, 0.0, 0.0, 0.0]],
            "correct_metric_val": 0,
        },
    ]

    evaluate_test_cases(test_cases, model, metrics, metric_name)


def test_precision_at_5():
    metric_name = "prec@5"
    k = 5
    model, metrics = build_metrics_test_model(
        bolt.train.metrics.PrecisionAtK, metric_name, {"k": k}
    )

    test_cases = [
        {
            "x": [[0.1, 0.1, 0.1, 0.1, 5.0]],
            "y": [[0.0, 0.0, 0.0, 0.0, 1.0]],
            "correct_metric_val": 0.2,
        },
        {
            "x": [[0.1, 0.1, 4.0, 0.1, 5.0]],
            "y": [[0.0, 0.0, 1.0, 0.0, 1.0]],
            "correct_metric_val": 0.4,
        },
        {
            "x": [[0.1, 0.1, 0.1, 1.0, 5.0]],
            "y": [[0.0, 0.0, 0.0, 0.0, 0.0]],
            "correct_metric_val": 0,
        },
        {
            "x": [[0.1, 0.1, 0.1, 1.0, 5.0]],
            "y": [[1.0, 1.0, 1.0, 0.0, 0.0]],
            "correct_metric_val": 0.6,
        },
    ]

    evaluate_test_cases(test_cases, model, metrics, metric_name)


def test_recall_at_1():
    metric_name = "rec@1"
    k = 1
    model, metrics = build_metrics_test_model(
        bolt.train.metrics.RecallAtK, metric_name, {"k": k}
    )

    test_cases = [
        {
            "x": [[0.1, 0.1, 0.1, 1.0, 5.0]],
            "y": [[0.0, 0.0, 0.0, 0.0, 1.0]],
            "correct_metric_val": 1,
        },
        {
            "x": [[0.1, 0.1, 1.1, 0.1, 5.0]],
            "y": [[0.0, 0.0, 1.0, 0.0, 1.0]],
            "correct_metric_val": 0.5,
        },
        {
            "x": [[0.1, 0.1, 0.1, 1.0, 5.0]],
            "y": [[1.0, 0.0, 0.0, 0.0, 0.0]],
            "correct_metric_val": 0,
        },
    ]

    evaluate_test_cases(test_cases, model, metrics, metric_name)


def test_recall_at_5():
    metric_name = "rec@5"
    k = 5
    model, metrics = build_metrics_test_model(
        bolt.train.metrics.RecallAtK, metric_name, {"k": k}
    )

    test_cases = [
        # Recall is technically undefined when tp+fn=0, but we return 0
        {
            "x": [[0.1, 0.1, 0.1, 1.0, 5.0]],
            "y": [[0.0, 0.0, 0.0, 0.0, 0.0]],
            "correct_metric_val": 0,
        },
        {
            "x": [[0.1, 0.1, 1.1, 1.0, 5.0]],
            "y": [[0.0, 0.0, 1.0, 0.0, 1.0]],
            "correct_metric_val": 1,
        },
        {
            "x": [[0.1, 0.1, 0.1, 1.0, 5.0]],
            "y": [[1.0, 1.0, 1.0, 1.0, 1.0]],
            "correct_metric_val": 1,
        },
    ]

    evaluate_test_cases(test_cases, model, metrics, metric_name)


# The following tie behavior tests are for documenting an unintuitive behavior
# with the p@k and r@k metrics, rather than testing for a desired/correct outcome
def test_precision_tie_behavior():
    metric_name = "prec@1"
    k = 1
    model, metrics = build_metrics_test_model(
        bolt.train.metrics.PrecisionAtK, metric_name, {"k": k}
    )

    test_cases = [
        {
            "x": [[1.0, 1.0, 1.0, 1.0, 1.0]],
            "y": [[0.0, 0.0, 0.0, 0.0, 1.0]],
            "correct_metric_val": 1,
        },
        {
            "x": [[1.0, 1.0, 1.0, 1.0, 1.0]],
            "y": [[1.0, 0.0, 0.0, 0.0, 0.0]],
            "correct_metric_val": 0,
        },
    ]

    evaluate_test_cases(test_cases, model, metrics, metric_name)


def test_recall_tie_behavior():
    metric_name = "rec@1"
    k = 1
    model, metrics = build_metrics_test_model(
        bolt.train.metrics.RecallAtK, metric_name, {"k": k}
    )

    test_cases = [
        {
            "x": [[1.0, 1.0, 1.0, 1.0, 1.0]],
            "y": [[0.0, 0.0, 0.0, 0.0, 1.0]],
            "correct_metric_val": 1,
        },
        {
            "x": [[1.0, 1.0, 1.0, 1.0, 1.0]],
            "y": [[1.0, 0.0, 0.0, 0.0, 0.0]],
            "correct_metric_val": 0,
        },
    ]

    evaluate_test_cases(test_cases, model, metrics, metric_name)


def harmonic_mean(prec, rec):
    return (2 * prec * rec) / (prec + rec)


def test_fmeasure():
    metric_name = "fmeasure"
    model, metrics = build_metrics_test_model(
        bolt.train.metrics.FMeasure, metric_name, {"threshold": 0.8}
    )

    test_cases = [
        {
            "x": [[0.85, 0.0, 0.9, 0.1, 0.4]],
            "y": [[0.0, 1.0, 1.0, 1.0, 1.0]],
            "correct_metric_val": harmonic_mean(prec=0.5, rec=0.25),
        },
        {
            "x": [[0.4, 0.9, 0.0, 1.0, 0.0]],
            "y": [[0.0, 0.0, 0.0, 1.0, 0.0]],
            "correct_metric_val": harmonic_mean(prec=0.5, rec=1.0),
        },
    ]

    evaluate_test_cases(test_cases, model, metrics, metric_name)


def mach_metric_test_helper(metric, expected_values):
    metric_name = "mach_metric"

    entity_to_hashes = {
        0: [0, 1, 2],
        1: [0, 1, 4],
        2: [2, 3, 4],
        3: [0, 2, 4],
        4: [1, 3, 4],
    }

    mach_index = dataset.MachIndex(
        entity_to_hashes=entity_to_hashes,
        output_range=LABEL_DIM,
        num_hashes=3,
    )

    model, metrics = build_metrics_test_model(
        metric,
        metric_name,
        {"mach_index": mach_index, "num_buckets_to_eval": 3, "k": 2},
    )

    test_cases = [
        {
            "x": [[1.0, 0.0, 1.0, 0.0, 0.0]],
            "y": [[1.0, 0.0, 0.0, 1.0, 1.0]],
            "correct_metric_val": expected_values[0],
        },
        {
            "x": [[1.0, 0.0, 1.0, 0.0, 0.0]],
            "y": [[0.0, 1.0, 0.0, 1.0, 0.0]],
            "correct_metric_val": expected_values[1],
        },
    ]

    evaluate_test_cases(test_cases, model, metrics, metric_name)


def test_mach_precision():
    mach_metric_test_helper(
        bolt.train.metrics.MachPrecision, expected_values=[1.0, 0.5]
    )


def test_mach_recall():
    mach_metric_test_helper(
        bolt.train.metrics.MachRecall, expected_values=[2.0 / 3.0, 0.5]
    )
