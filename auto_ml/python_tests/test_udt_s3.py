import os

import boto3
import pytest
from download_datasets import download_census_income
from model_test_utils import compute_evaluate_accuracy, train_udt_census_income
from moto import mock_s3

pytestmark = [pytest.mark.unit, pytest.mark.release]

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


def setup_census_on_s3(
    s3, local_census_file_path, target_census_bucket, target_census_key
):
    s3.create_bucket(Bucket=target_census_bucket)
    s3.upload_file(local_census_file_path, target_census_bucket, target_census_key)


@mock_s3
def test_utd_census_income_s3(download_census_income, s3):
    s3_bucket = "test_bucket"
    s3_key = "path/to/census/data.csv"
    local_train_file, local_test_file, inference_samples = download_census_income
    setup_census_on_s3(
        s3,
        local_census_file_path=local_train_file,
        target_census_bucket=s3_bucket,
        target_census_key=s3_key,
    )
    s3_path = f"s3://{s3_bucket}/{s3_key}"
    model = train_udt_census_income(s3_path)

    acc = compute_evaluate_accuracy(
        model, local_test_file, inference_samples, use_class_name=True
    )
    assert acc >= 0.8
