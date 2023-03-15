import boto3
import pytest
from moto import mock_s3
from moto.server import ThreadedMotoServer
from telemetry_testing_utils import run_udt_telemetry_test
from thirdai import telemetry

pytestmark = [pytest.mark.unit, pytest.mark.release]

THIRDAI_TEST_TELEMETRY_S3_DIR = "s3://test_bucket/test_telemetry_dir"
THIRDAI_TEST_TELEMETRY_BUCKET = "test_bucket"

MOTO_SERVER_PORT = 20732


@pytest.fixture(scope="module", autouse=True)
def moto_server_fixture():
    server = ThreadedMotoServer(port=MOTO_SERVER_PORT)
    server.start()
    yield
    return server


@mock_s3
def test_udt_telemetry_s3():
    s3 = boto3.client("s3")
    s3.create_bucket(Bucket=THIRDAI_TEST_TELEMETRY_BUCKET)
    s3_path = telemetry.start(
        write_dir=THIRDAI_TEST_TELEMETRY_S3_DIR,
        optional_endpoint_url=f"http://127.0.0.1:{MOTO_SERVER_PORT}",
    )
    run_udt_telemetry_test(telemetry_start_method=("s3", s3_path))


@mock_s3
def test_telemetry_bad_s3_file():
    with pytest.raises(
        ValueError, match="Telemetry process terminated early with exit code 1"
    ):
        telemetry.start(
            write_dir="s3://this/does/not/exist",
            optional_endpoint_url=f"http://127.0.0.1:{MOTO_SERVER_PORT}",
        )
