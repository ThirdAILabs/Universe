from typing import Tuple
from urllib.parse import urlparse

import boto3
import requests
from prometheus_client.parser import text_string_to_metric_families

# This line uses a hack where we can import functions from different test files
# as long as this file is run from bin/python-format.sh. To run the test files
# in this directory, you should run a command in the following format:
# bin/python-test.sh -k "test_basic_telemetry"
from test_udt_simple import (
    TEST_FILE,
    batch_sample,
    make_simple_trained_model,
    single_sample,
)
from thirdai import telemetry

THIRDAI_TEST_TELEMETRY_PORT = 20730


def scrape_telemetry(telemetry_start_method):
    telemetry = {}
    if telemetry_start_method[0] == "port":
        raw_telemetry = requests.get(telemetry_start_method[1]).content.decode("utf-8")
    elif telemetry_start_method[0] == "file":
        with open(telemetry_start_method[1]) as f:
            raw_telemetry = f.read()
    elif telemetry_start_method[0] == "s3":
        client = boto3.client("s3")
        parsed_s3_path = urlparse(telemetry_start_method[1])
        raw_telemetry = (
            # We need to remove the first char of the parsed_s3_path because
            # urlparse keeps a leading slash, but boto3/moto expects keys
            # not to have leading slashes.
            client.get_object(
                Bucket=parsed_s3_path.netloc,
                Key=parsed_s3_path.path[1:],
            )["Body"]
            .read()
            .decode("utf-8")
        )
    else:
        raise ValueError(f"Unknown method {telemetry_start_method}")
    for family in text_string_to_metric_families(raw_telemetry):
        for name, labels, value, _, _ in family.samples:
            if name not in telemetry:
                telemetry[name] = []
            telemetry[name].append((labels, value))
    return telemetry


# The telemetry dictionary is a map from telemetry names to a list of
# "metric recordings", where each metric recording is a tuple of 1. a
# dictionary of labels and 2. a metric value. This function works on count or
# sum metrics, which will have just one metric recording, and returns the scalar
# metric value.
def get_metric_value(telemetry_dict, key):
    # We assert that the passed in key corresponds to a metric name with just a
    # single set of valid labels,  as specified in the function comment.
    assert len(telemetry_dict[key]) == 1
    return telemetry_dict[key][0][1]


# This will also stop telemetry by the end of the method
# telemetry_start_method should be a tuple of the telemetry destination we are
# testing (that it must have been started with) and the identifier
# returned from telemetry.start(). Valid method names are port, file, and s3.
def run_udt_telemetry_test(telemetry_start_method: Tuple[str, str]):
    import time

    # For push based tests (files and s3), we need to call telemetry.stop()
    # after we finish the udt calls to make sure the writing is flushed. Forbin/
    # the port check test, we instead want to wait until the end of the method
    # to kill telemetry, since the prometheus client will only host telemetry
    # on the port until stop is called.
    kill_telemetry_after_udt = telemetry_start_method[0] != "port"

    eval_count = 2
    explain_count = 10
    predict_count = 20
    batch_predict_count = 15

    train_start = time.time()
    udt_model = make_simple_trained_model(numerical_temporal=False)
    train_duration = time.time() - train_start

    eval_start = time.time()
    for _ in range(eval_count):
        udt_model.evaluate(TEST_FILE)
        udt_model.reset_temporal_trackers()
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

    if kill_telemetry_after_udt:
        telemetry.stop()

    scraped_telemetry = scrape_telemetry(telemetry_start_method)

    assert (
        get_metric_value(
            scraped_telemetry, "thirdai_udt_training_duration_seconds_count"
        )
        == 1
    )
    assert (
        get_metric_value(scraped_telemetry, "thirdai_udt_training_duration_seconds_sum")
        <= train_duration
    )

    assert (
        get_metric_value(
            scraped_telemetry, "thirdai_udt_explanation_duration_seconds_count"
        )
        == explain_count
    )
    assert (
        get_metric_value(
            scraped_telemetry, "thirdai_udt_explanation_duration_seconds_sum"
        )
        <= explain_duration
    )

    assert (
        get_metric_value(
            scraped_telemetry, "thirdai_udt_evaluation_duration_seconds_count"
        )
        == eval_count
    )
    assert (
        get_metric_value(
            scraped_telemetry, "thirdai_udt_evaluation_duration_seconds_sum"
        )
        <= eval_duration
    )

    assert (
        get_metric_value(
            scraped_telemetry, "thirdai_udt_prediction_duration_seconds_count"
        )
        == predict_count
    )
    assert (
        get_metric_value(
            scraped_telemetry, "thirdai_udt_prediction_duration_seconds_sum"
        )
        <= predict_duration
    )

    # We need to multiply by batch_sample_size because we add one entry
    # to the Prometheus histogram for every item in the batch
    assert (
        get_metric_value(
            scraped_telemetry, "thirdai_udt_batch_prediction_duration_seconds_count"
        )
        == batch_predict_count * batch_sample_size
    )

    # We need to multiply by batch_sample_size because we add one entry
    # to the Prometheus histogram for every item in the batch
    assert (
        get_metric_value(
            scraped_telemetry, "thirdai_udt_batch_prediction_duration_seconds_sum"
        )
        <= batch_predict_duration * batch_sample_size
    )

    telemetry.stop()
