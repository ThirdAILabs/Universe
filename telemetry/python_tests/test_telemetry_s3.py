import pytest
from moto import mock_s3
from moto.server import ThreadedMotoServer
from telemetry_testing_utils import run_udt_telemetry_test
from thirdai import telemetry

pytestmark = [pytest.mark.unit, pytest.mark.release]

THIRDAI_TEST_TELEMETRY_S3_DIR = "s3://test_bucket/test_telemetry_dir"


@mock_s3
def test_udt_telemetry_s3():
    server = ThreadedMotoServer(port=20731)
    try:
        server.start()
        s3_path = telemetry.start(
            write_dir=THIRDAI_TEST_TELEMETRY_S3_DIR,
            optional_endpoint_url="http://127.0.0.1:5000",
        )
        run_udt_telemetry_test(method=("s3", s3_path), kill_telemetry_after_udt=True)
    finally:
        telemetry.stop()
        server.stop()
