import os
import subprocess
import uuid
from os import getenv, path

import pytest
from google.cloud import storage

pytestmark = [pytest.mark.unit, pytest.mark.release]


os.environ["GCP_PROJECT"] = "udt-model-testing"
os.environ["BUCKET"] = "udt-testing-bucket"


@pytest.fixture(scope="module")
def storage_client():
    client = storage.Client()
    client.create_bucket(bucket_or_name=os.getenv("BUCKET"), location="us-west1")
    yield client


@pytest.fixture(scope="module")
def bucket_object(storage_client):
    bucket_object = storage_client.get_bucket(os.getenv("BUCKET"))
    yield bucket_object


@pytest.fixture(scope="module")
def uploaded_file(bucket_objet, file_name):
    blob = bucket_object.blob(file_name)

    test_dir = path.dirname(path.abspath(__file__))
    blob.upload_from_filename(path.join(test_dir, file_name))
    yield file_name
    blob.delete()


@pytest.fixture(scope="module")
def prepare_data(bucket_object, download_census_income):
    train_file, test_file, inference_batch = download_census_income

    train_blob = bucket_object.blob(train_file)
    test_blob = bucket_object.blob(test_file)

    testing_dir = path.dirname(path.abspath(__file__))
    train_blob.upload_from_filename(path.join(testing_dir, train_file))
    test_blob.upload_from_filename(path.join(testing_dir, test_file))

    yield train_file


def test_udt(prepare_data):
    model = get_udt_census_income_model()
    train_file = prepare_data
    model.train(train_file, epochs=5, learning_rate=0.01)
