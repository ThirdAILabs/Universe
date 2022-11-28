import pytest
from thirdai import metrics

THIRDAI_TEST_METRICS_PORT = 20730

@pytest.fixture(autouse=True)
def with_metrics():
    metrics.start_metrics(THIRDAI_TEST_METRICS_PORT)
    yield
    metrics.stop_metrics()

def test_train_metrics():
    pass

def test_predict_metrics():
    pass
