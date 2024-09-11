import math
import os
import sys

import boto3
import pytest
from moto import mock_aws

pytestmark = [pytest.mark.unit, pytest.mark.release]

batch_size = 64
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
    with mock_aws():
        yield boto3.client("s3", region_name="us-east-1")


def setup_mock_aws(s3):
    s3.create_bucket(Bucket="test_bucket")
    s3.put_object(
        Bucket=bucket_name,
        Key="find/numbers/ones",
        Body="\n".join(["1"] * num_lines_per_file),
    )


def load_all_batches(storage_path, batch_size):
    from thirdai import dataset

    source = dataset.CSVDataSource(storage_path=storage_path)
    batches = []
    while True:
        next_batch = source.next_batch(batch_size)
        if not next_batch:
            break
        batches.append(next_batch)
    return batches


def load_all_lines(storage_path, batch_size):
    from thirdai import dataset

    source = dataset.CSVDataSource(storage_path=storage_path)
    lines = []
    while True:
        next_line = source.next_line()
        if not next_line:
            break
        lines.append(next_line)
    return lines


# Mac wheel builds are failing with the error:
# AttributeError: 'MockRawResponse' object has no attribute 'raw_headers'
@pytest.mark.skipif(sys.platform == "darwin")
@mock_aws
def test_s3_data_source_by_batch(s3):
    setup_mock_aws(s3)

    batches = load_all_batches(
        storage_path=f"s3://{bucket_name}/find/numbers/ones", batch_size=batch_size
    )

    assert len(batches) == math.ceil(num_lines_per_file / batch_size)

    for batch in batches[:-1]:
        assert len(batch) == batch_size

    assert len(batches[-1]) == num_lines_per_file % batch_size

    # check that the content matches what's expected
    assert "".join(["".join(batch) for batch in batches]) == "1" * num_lines_per_file


# This is similar to test_s3_data_source_by_batch, but additionally ensures
# that the linewise loading is correct
# Mac wheel builds are failing with the error:
# AttributeError: 'MockRawResponse' object has no attribute 'raw_headers'
@pytest.mark.skipif(sys.platform == "darwin")
@mock_aws
def test_s3_data_source_by_line(s3):
    setup_mock_aws(s3)

    lines = load_all_lines(
        storage_path=f"s3://{bucket_name}/find/numbers/ones", batch_size=batch_size
    )
    assert len(lines) == num_lines_per_file
    assert "".join(line for line in lines) == "1" * num_lines_per_file
