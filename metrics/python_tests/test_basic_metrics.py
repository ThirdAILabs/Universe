import pytest
import requests
from prometheus_client.parser import text_string_to_metric_families

# This line uses a hack where we can import functions from different test files
# as long as this file is run from bin/python-format.sh. To run just this file,
# run bin/python-test.sh -k "test_basic_metrics"
from test_udt_simple import make_simple_trained_model
from thirdai import metrics

THIRDAI_TEST_METRICS_PORT = 20730
THIRDAI_METRICS_URL = f"http://localhost:{20730}/metrics"


@pytest.fixture(autouse=True)
def with_metrics():
    metrics.start_metrics(THIRDAI_TEST_METRICS_PORT)
    yield
    metrics.stop_metrics()


def scrape_metrics(url):
    metrics = {}
    raw_metrics = requests.get(url).content.decode("utf-8")
    for family in text_string_to_metric_families(raw_metrics):
        for name, labels, value, _, _ in family.samples:
            if name not in metrics:
                metrics[name] = []
            metrics[name].append((name, labels, value))
    return metrics


def test_train_metrics():
    make_simple_trained_model()

    for name, data in scrape_metrics(THIRDAI_METRICS_URL).items():
        print(name, data)


def test_predict_metrics():
    pass
