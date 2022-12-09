import os
import subprocess
from os import getenv, path

import mock
import numpy as np
import pandas as pd
import pytest
from google.cloud import storage
from thirdai import dataset

pytestmark = [pytest.mark.unit, pytest.mark.release]


BUCKET = "testing-bucket"
BLOB = "storage-object-name"
TEST_FILE = "test_file.csv"
GCS_CREDENTIALS = "test_credentials.json"
TEST_DATASET_SIZE = 200


def clean_up():
    if os.path.exists(TEST_FILE):
        os.remove(TEST_FILE)


@pytest.fixture(scope="module")
def testing_dataframe():
    return pd.DataFrame(np.random.randint(0, 100, size=(TEST_DATASET_SIZE, 2)))


@pytest.fixture(scope="module")
def disk_persisted_dataset(testing_dataframe):
    testing_dataframe.to_csv(TEST_FILE)


@mock.patch("pandas.read_csv")
def test_csv_loader_from_gcs(
    pandas_read_csv, disk_persisted_dataset, testing_dataframe
):
    """
    This unit test uses a mock for pandas.read_csv function because otherwise the actual
    function call in the CSVDataLoader class will attempt to establish a connection
    with the given mock storage path for GCS, which will throw a network Error.
    """
    storage_client = mock.create_autospec(storage.Client)
    bucket = storage_client.create_bucket(bucket_or_name=BUCKET, location="us-west1")
    blob = bucket.blob(BLOB)

    blob.upload_from_filename(TEST_FILE)
    pandas_read_csv.return_value = testing_dataframe.iterrows()
    blob.upload_from_filename.assert_called_with(TEST_FILE)

    # create a csv data loader
    loader = dataset.CSVDataLoader(
        storage_path=f"gcs://{BUCKET}",
        batch_size=5,
    )

    all_records = []
    while True:
        record = loader.next_line()
        if record == None:
            break
        all_records.append(record[1])

    assert len(all_records) == TEST_DATASET_SIZE
    assert testing_dataframe.equals(pd.concat(all_records, axis=1).T)

    clean_up()
