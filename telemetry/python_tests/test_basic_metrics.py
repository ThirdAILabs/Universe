import pytest
import requests
from prometheus_client.parser import text_string_to_metric_families

# This line uses a hack where we can import functions from different test files
# as long as this file is run from bin/python-format.sh. To run just this file,
# run bin/python-test.sh -k "test_basic_metrics"
from test_udt_simple import (
    TEST_FILE,
    batch_sample,
    make_simple_trained_model,
    single_sample,
)
from thirdai import telemetry

THIRDAI_TEST_METRICS_PORT = 20730
THIRDAI_TEST_METRICS_URL = f"http://localhost:{20730}/metrics"


# autouse=True means that every test in this file will require this fixture
# and start and stop metrics, ensuring that the test can check metrics
# independent of other tests.
@pytest.fixture(autouse=True)
def with_metrics():
    telemetry.start(THIRDAI_TEST_METRICS_PORT)
    # Yielding here means that telemetry.stop() will get called after the
    # test finishes, see
    # https://docs.pytest.org/en/6.2.x/fixture.html#yield-fixtures-recommended
    yield
    telemetry.stop()


def scrape_metrics(url):
    metrics = {}
    raw_metrics = requests.get(url).content.decode("utf-8")
    for family in text_string_to_metric_families(raw_metrics):
        for name, labels, value, _, _ in family.samples:
            if name not in metrics:
                metrics[name] = []
            metrics[name].append((labels, value))
    return metrics


def get_count(metrics_dict, key):
    assert len(metrics_dict[key]) == 1
    return metrics_dict[key][0][1]


def check_metrics(
    metrics,
    train_count,
    train_duration,
    eval_count,
    eval_duration,
    explain_count,
    explain_duration,
    predict_count,
    predict_duration,
    batch_predict_count,
    batch_predict_duration,
):
    assert (
        get_count(metrics, "thirdai_udt_training_duration_seconds_count") == train_count
    )
    assert (
        get_count(metrics, "thirdai_udt_training_duration_seconds_sum")
        <= train_duration
    )

    assert (
        get_count(metrics, "thirdai_udt_explanation_duration_seconds_count")
        == explain_count
    )
    assert (
        get_count(metrics, "thirdai_udt_explanation_duration_seconds_sum")
        <= explain_duration
    )

    assert (
        get_count(metrics, "thirdai_udt_evaluation_duration_seconds_count")
        == eval_count
    )
    assert (
        get_count(metrics, "thirdai_udt_evaluation_duration_seconds_sum")
        <= eval_duration
    )

    assert (
        get_count(metrics, "thirdai_udt_prediction_duration_seconds_count")
        == predict_count
    )
    assert (
        get_count(metrics, "thirdai_udt_prediction_duration_seconds_sum")
        <= predict_duration
    )

    assert (
        get_count(metrics, "thirdai_udt_batch_prediction_duration_seconds_count")
        == batch_predict_count
    )
    assert (
        get_count(metrics, "thirdai_udt_batch_prediction_duration_seconds_sum")
        <= batch_predict_duration
    )


def test_udt_metrics():
    import time

    eval_count = 2
    explain_count = 10
    predict_count = 20
    batch_predict_count = 15

    train_start = time.time()
    udt_model = make_simple_trained_model()
    train_duration = time.time() - train_start

    eval_start = time.time()
    for _ in range(eval_count):
        udt_model.evaluate(TEST_FILE)
    eval_duration = time.time() - eval_start

    explain_start = time.time()
    for _ in range(explain_count):
        udt_model.explain(single_sample(), target_class="1")
    explain_duration = time.time() - explain_start

    predict_start = time.time()
    for _ in range(predict_count):
        udt_model.predict(single_sample())
    predict_duration = time.time() - predict_start

    batch_predict_start = time.time()
    batch_sample_size = len(batch_sample())
    for _ in range(batch_predict_count):
        udt_model.predict_batch(batch_sample())
    batch_predict_duration = time.time() - batch_predict_start

    metrics = scrape_metrics(THIRDAI_TEST_METRICS_URL)
    check_metrics(
        metrics,
        train_count=1,
        train_duration=train_duration,
        eval_count=eval_count,
        eval_duration=eval_duration,
        explain_count=explain_count,
        explain_duration=explain_duration,
        predict_count=predict_count,
        predict_duration=predict_duration,
        # We need to multiply by batch_sample_size because we add one entry
        # to the Prometheus histogram for every item in the batch
        batch_predict_count=batch_predict_count * batch_sample_size,
        batch_predict_duration=batch_predict_duration * batch_sample_size,
    )


def test_error_starting_two_metric_clients():
    with pytest.raises(
        RuntimeError,
        match="Trying to start telemetry client when one is already running.*",
    ):
        telemetry.start()


def test_stop_and_start_metrics():
    telemetry.stop()
    telemetry.start(THIRDAI_TEST_METRICS_PORT)
