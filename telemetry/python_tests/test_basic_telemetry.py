import botocore
import pytest
from telemetry_testing_utils import THIRDAI_TEST_TELEMETRY_PORT, run_udt_telemetry_test
from thirdai import telemetry

pytestmark = [pytest.mark.unit, pytest.mark.unit]

THIRDAI_TEST_TELEMETRY_DIR = "test_telemetry_dir"


def test_udt_telemetry_port():
    telemetry_url = telemetry.start(port=THIRDAI_TEST_TELEMETRY_PORT)
    run_udt_telemetry_test(telemetry_start_method=("port", telemetry_url))


def test_udt_telemetry_file():
    file = telemetry.start(
        port=THIRDAI_TEST_TELEMETRY_PORT, write_dir=THIRDAI_TEST_TELEMETRY_DIR
    )
    run_udt_telemetry_test(telemetry_start_method=("file", file))


def test_error_starting_two_telemetry_clients():
    telemetry.start(port=THIRDAI_TEST_TELEMETRY_PORT)
    with pytest.raises(
        RuntimeError,
        match="Trying to start telemetry client when one is already running.*",
    ):
        telemetry.start(port=THIRDAI_TEST_TELEMETRY_PORT + 1)
    telemetry.stop()


def test_stop_and_start_telemetry():
    """
    This ensures that we can start and stop multiple times without messing
    up the telemetry state and throwing an error
    """
    for _ in range(2):
        telemetry.start(port=THIRDAI_TEST_TELEMETRY_PORT)
        telemetry.stop()
    for _ in range(2):
        telemetry.start(
            port=THIRDAI_TEST_TELEMETRY_PORT, write_dir=THIRDAI_TEST_TELEMETRY_DIR
        )
        telemetry.stop()


def test_bad_udt_telemetry_file():
    with pytest.raises(ValueError, match="this://should/not/work"):
        telemetry.start(write_dir="this://should/not/work")
    telemetry.stop()
