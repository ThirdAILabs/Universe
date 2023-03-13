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


def scrape_telemetry(method):
    telemetry = {}
    if method[0] == "port":
        raw_telemetry = requests.get(method[1]).content.decode("utf-8")
    elif method[0] == "file":
        with open(method[1]) as f:
            raw_telemetry = f.read()
    else:
        raise ValueError(f"Unknown method {method}")
    for family in text_string_to_metric_families(raw_telemetry):
        for name, labels, value, _, _ in family.samples:
            if name not in telemetry:
                telemetry[name] = []
            telemetry[name].append((labels, value))
    return telemetry


def get_count(telemetry_dict, key):
    assert len(telemetry_dict[key]) == 1
    return telemetry_dict[key][0][1]


def check_telemetry(
    telemetry,
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
        get_count(telemetry, "thirdai_udt_training_duration_seconds_count")
        == train_count
    )
    assert (
        get_count(telemetry, "thirdai_udt_training_duration_seconds_sum")
        <= train_duration
    )

    assert (
        get_count(telemetry, "thirdai_udt_explanation_duration_seconds_count")
        == explain_count
    )
    assert (
        get_count(telemetry, "thirdai_udt_explanation_duration_seconds_sum")
        <= explain_duration
    )

    assert (
        get_count(telemetry, "thirdai_udt_evaluation_duration_seconds_count")
        == eval_count
    )
    assert (
        get_count(telemetry, "thirdai_udt_evaluation_duration_seconds_sum")
        <= eval_duration
    )

    assert (
        get_count(telemetry, "thirdai_udt_prediction_duration_seconds_count")
        == predict_count
    )
    assert (
        get_count(telemetry, "thirdai_udt_prediction_duration_seconds_sum")
        <= predict_duration
    )

    assert (
        get_count(telemetry, "thirdai_udt_batch_prediction_duration_seconds_count")
        == batch_predict_count
    )
    assert (
        get_count(telemetry, "thirdai_udt_batch_prediction_duration_seconds_sum")
        <= batch_predict_duration
    )


# kill_telemetry_after_udt should be true if we are writing to a file or s3,
# since we want to make sure the writing is flushed to the file so we do not
# read a partial read. It should be false (and killed manually) if we are just
# checking the port, since that will only be available as long as telemetry is
# running.
def run_udt_telemetry_test(method, kill_telemetry_after_udt):
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

    if kill_telemetry_after_udt:
        telemetry.stop()

    scraped_telemetry = scrape_telemetry(method)
    check_telemetry(
        scraped_telemetry,
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
