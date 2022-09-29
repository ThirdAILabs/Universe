import boto3
from moto import mock_s3
import os
import pytest
import math

pytestmark = [pytest.mark.unit, pytest.mark.release]

batch_size = 64
# Number of lines in each mock s3 file we will create (we create 4, see setup_mock_s3)
num_lines_per_file = 100
# We will have 4 files in the bucket, but should only parse batches from the
# 3 whose keys start with find
total_num_lines_to_return = num_lines_per_file * 3

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
        Bucket="test_bucket",
        Key="find/numbers/zeros",
        Body="\n".join(["0"] * num_lines_per_file),
    )
    s3.put_object(
        Bucket="test_bucket",
        Key="find/numbers/ones",
        Body="\n".join(["1"] * num_lines_per_file),
    )
    s3.put_object(
        Bucket="test_bucket",
        Key="find/letters/ds",
        Body="\n".join(["d"] * num_lines_per_file),
    )
    s3.put_object(
        Bucket="test_bucket",
        Key="dontfind/test",
        Body="\n".join(["X"] * num_lines_per_file),
    )


def load_all_batches(bucket_name, prefix_filter, batch_size):
    from thirdai import dataset

    loader = dataset.S3DataLoader(
        bucket_name=bucket_name, prefix_filter=prefix_filter, batch_size=batch_size
    )
    batches = []
    while True:
        next_batch = loader.next_batch()
        if not next_batch:
            break
        batches.append(next_batch)
    return batches


def load_all_lines(bucket_name, prefix_filter, batch_size):
    from thirdai import dataset

    loader = dataset.S3DataLoader(
        bucket_name=bucket_name, prefix_filter=prefix_filter, batch_size=batch_size
    )
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

    batches = load_all_batches(
        bucket_name="test_bucket", prefix_filter="find", batch_size=batch_size
    )

    assert len(batches) == math.ceil(total_num_lines_to_return / batch_size)

    for batch in batches[:-1]:
        assert len(batch) == batch_size
    assert len(batches[-1]) == total_num_lines_to_return % batch_size

    concatenated = "".join(["".join(batch) for batch in batches])
    assert concatenated == "d" * 100 + "1" * 100 + "0" * 100


# This is similar to test_s3_data_loader_by_batch, but additionally ensures
# that the linewise loading is correct
@mock_s3
def test_s3_data_loader_by_line(s3):

    setup_mock_s3(s3)

    lines = load_all_lines(
        bucket_name="test_bucket", prefix_filter="find", batch_size=batch_size
    )

    assert len(lines) == total_num_lines_to_return

    concatenated = "".join(line for line in lines)
    assert concatenated == "d" * 100 + "1" * 100 + "0" * 100
