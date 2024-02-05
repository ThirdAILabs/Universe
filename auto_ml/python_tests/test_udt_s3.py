import os

import boto3
import pytest
from download_dataset_fixtures import download_census_income
from model_test_utils import compute_evaluate_accuracy, get_udt_census_income_model
from moto import mock_aws
from thirdai import bolt

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
    with mock_aws():
        yield boto3.client("s3", region_name="us-east-1")


def setup_census_on_s3(s3, local_train_file, local_test_file):
    s3_bucket = "test_bucket"
    s3_train_key = "path/to/census/data.csv"
    s3_test_key = "path/to/test/data.csv"
    s3_test_path = f"s3://{s3_bucket}/{s3_test_key}"
    s3_train_path = f"s3://{s3_bucket}/{s3_train_key}"

    s3.create_bucket(Bucket=s3_bucket)
    s3.upload_file(local_train_file, s3_bucket, s3_train_key)
    s3.upload_file(local_test_file, s3_bucket, s3_test_key)

    return s3_train_path, s3_test_path


ACCURACY_THRESHOLD = 0.8


def train_and_evaluate(model_to_test, train_path, test_path, inference_samples):
    model_to_test.train(train_path, epochs=5, learning_rate=0.01)
    acc = compute_evaluate_accuracy(model_to_test, test_path)
    assert acc >= ACCURACY_THRESHOLD


@mock_aws
def test_udt_census_income_s3(download_census_income, s3):
    local_train_file, local_test_file, inference_samples = download_census_income
    s3_train_path, s3_test_path = setup_census_on_s3(
        s3, local_train_file, local_test_file
    )

    model = get_udt_census_income_model()
    train_and_evaluate(model, s3_train_path, s3_test_path, inference_samples)

    # Save, load, train, and evaluate on s3 again to make sure that the extra
    # bound python methods stick around
    save_path = "census_udt.model"
    model.save(save_path)
    loaded_model = bolt.UniversalDeepTransformer.load(save_path)
    os.remove(save_path)

    train_and_evaluate(loaded_model, s3_train_path, s3_test_path, inference_samples)
