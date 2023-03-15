import pytest
from telemetry_testing_utils import run_udt_telemetry_test
from thirdai import telemetry

pytestmark = [pytest.mark.unit, pytest.mark.release]

THIRDAI_TEST_TELEMETRY_PORT = 20730
THIRDAI_TEST_TELEMETRY_DIR = "test_telemetry_dir"


def test_udt_telemetry_port():
    telemetry_url = telemetry.start(port=THIRDAI_TEST_TELEMETRY_PORT)
    run_udt_telemetry_test(telemetry_start_method=("port", telemetry_url))


def test_udt_telemetry_file():
    file = telemetry.start(write_dir=THIRDAI_TEST_TELEMETRY_DIR)
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
    telemetry.start(port=THIRDAI_TEST_TELEMETRY_PORT)
    telemetry.stop()
    telemetry.start(port=THIRDAI_TEST_TELEMETRY_PORT)
    telemetry.stop()


def test_bad_udt_telemetry_file():
    telemetry.start(write_dir="this_should://def/not/work")
