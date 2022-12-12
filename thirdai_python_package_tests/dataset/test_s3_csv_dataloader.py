import math
import os

import boto3
import pytest
from moto import mock_s3

pytestmark = [pytest.mark.unit, pytest.mark.release]

batch_size = 64
# Number of lines in each mock s3 file we will create (we create 4, see setup_mock_s3)
num_lines_per_file = 100
bucket_name = "test_bucket"

# These fixtures allow us to take in a parameter to our tests called "s3" that
# ensures all s3 calls will not call "actual" s3
# See http://docs.getmoto.org/en/latest/docs/getting_started.html#example-on-usage
@pytest.fixture(scope="function")
def aws_credentials():
    """Mocked AWS Credentials for moto."""
    # os.environ["MOTO_ALLOW_NONEXISTENT_REGION"] = "True"
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"


@pytest.fixture(scope="function")
def s3(aws_credentials):
    with mock_s3():
        yield boto3.client("s3", region_name="us-east-1")


def setup_mock_s3(s3):
    s3.create_bucket(Bucket="test_bucket")
    s3.put_object(
        Bucket=bucket_name,
        Key="find/numbers/zeros",
        Body="\n".join(["0"] * num_lines_per_file),
    )
    s3.put_object(
        Bucket=bucket_name,
        Key="find/numbers/ones",
        Body="\n".join(["1"] * num_lines_per_file),
    )
    s3.put_object(
        Bucket=bucket_name,
        Key="find/letters/ds",
        Body="\n".join(["d"] * num_lines_per_file),
    )
    s3.put_object(
        Bucket=bucket_name,
        Key="dontfind/test",
        Body="\n".join(["X"] * num_lines_per_file),
    )


def key_map():
    return {"ones": "1", "ds": "d", "zeros": "0"}


def load_all_batches(storage_path, batch_size):
    from thirdai import dataset

    loader = dataset.CSVDataLoader(storage_path=storage_path, batch_size=batch_size)
    batches = []
    while True:
        next_batch = loader.next_batch()
        if not next_batch:
            break
        batches.append(next_batch)
    return batches


def load_all_lines(storage_path, batch_size):
    from thirdai import dataset

    loader = dataset.CSVDataLoader(storage_path=storage_path, batch_size=batch_size)
    lines = []
    while True:
        next_line = loader.next_line()
        if not next_line:
            break
        lines.append(next_line)
    return lines


# This test sets up a mock S3 bucket using moto, puts mock objects in the
# bucket, and then ensures that the loader returns all of the lines from the
# objects that follow the prefix in the correct batches.
@mock_s3
def test_s3_data_loader_by_batch(s3):
    setup_mock_s3(s3)

    for key in s3.list_objects_v2(Bucket="test_bucket", Prefix="find")["Contents"]:
        object = key["Key"]
        batches = load_all_batches(
            storage_path=f"s3://{bucket_name}/{object}", batch_size=batch_size
        )

        assert len(batches) == math.ceil(num_lines_per_file / batch_size)

        for batch in batches[:-1]:
            assert len(batch) == batch_size

        assert len(batches[-1]) == num_lines_per_file % batch_size

        # check that the content matches what's expected
        assert (
            "".join(["".join(batch) for batch in batches])
            == key_map()[object.split("/")[-1]] * num_lines_per_file
        )


# This is similar to test_s3_data_loader_by_batch, but additionally ensures
# that the linewise loading is correct
@mock_s3
def test_s3_data_loader_by_line(s3):

    setup_mock_s3(s3)

    for key in s3.list_objects_v2(Bucket="test_bucket", Prefix="find")["Contents"]:
        object = key["Key"]
        lines = load_all_lines(
            storage_path=f"s3://{bucket_name}/{object}", batch_size=batch_size
        )
        assert len(lines) == num_lines_per_file
        assert (
            "".join(line for line in lines)
            == key_map()[object.split("/")[-1]] * num_lines_per_file
        )
