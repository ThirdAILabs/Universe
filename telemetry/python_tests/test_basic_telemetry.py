import pytest
from telemetry_testing_utils import run_udt_telemetry_test
from thirdai import telemetry

pytestmark = [pytest.mark.unit, pytest.mark.release]

THIRDAI_TEST_TELEMETRY_PORT = 20730
THIRDAI_TEST_TELEMETRY_DIR = "test_telemetry_dir"


def test_udt_telemetry_port():
    try:
        telemetry_url = telemetry.start(port=THIRDAI_TEST_TELEMETRY_PORT)
        run_udt_telemetry_test(
            method=("port", telemetry_url), kill_telemetry_after_udt=False
        )
    finally:
        telemetry.stop()


def test_udt_telemetry_file():
    try:
        file = telemetry.start(write_dir=THIRDAI_TEST_TELEMETRY_DIR)
        run_udt_telemetry_test(method=("file", file), kill_telemetry_after_udt=True)
    finally:
        telemetry.stop()


def test_error_starting_two_telemetry_clients():
    try:
        telemetry.start(port=THIRDAI_TEST_TELEMETRY_PORT)
        with pytest.raises(
            RuntimeError,
            match="Trying to start telemetry client when one is already running.*",
        ):
            telemetry.start(port=THIRDAI_TEST_TELEMETRY_PORT + 1)
    finally:
        telemetry.stop()


def test_stop_and_start_telemetry():
    try:
        telemetry.start(port=THIRDAI_TEST_TELEMETRY_PORT)
        telemetry.stop()
        telemetry.start(port=THIRDAI_TEST_TELEMETRY_PORT)
    finally:
        telemetry.stop()
