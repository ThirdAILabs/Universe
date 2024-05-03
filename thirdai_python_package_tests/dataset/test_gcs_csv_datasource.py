import os
import subprocess
from os import getenv, path

import mock
import numpy as np
import pandas as pd
import pytest
from google.cloud import storage
from thirdai import dataset

pytestmark = [pytest.mark.unit, pytest.mark.unit]


BUCKET = "testing-bucket"
BLOB = "storage-object-name"
TEST_FILE = "test_file.csv"
TEST_DATASET_SIZE = 10
NUMBER_OF_COLS = 2


def clean_up():
    if os.path.exists(TEST_FILE):
        os.remove(TEST_FILE)


@pytest.fixture(scope="module")
def testing_dataframe():
    dataframe = pd.DataFrame(
        np.random.randint(0, 100, size=(TEST_DATASET_SIZE, NUMBER_OF_COLS))
    )
    return dataframe.applymap(str)


@pytest.fixture(scope="module")
def disk_persisted_dataset(testing_dataframe):
    testing_dataframe.to_csv(TEST_FILE)


@mock.patch("pandas.read_csv")
def test_csv_source_from_gcs(
    pandas_read_csv, disk_persisted_dataset, testing_dataframe
):
    """
    This unit test uses a mock for pandas.read_csv function because otherwise the actual
    function call in the CSVDataSource class will attempt to establish a connection
    with the given storage path for GCS, which will throw a network Error.
    """
    storage_client = mock.create_autospec(storage.Client)
    bucket = storage_client.create_bucket(bucket_or_name=BUCKET, location="us-west1")
    blob = bucket.blob(BLOB)

    blob.upload_from_filename(TEST_FILE)
    pandas_read_csv.return_value = testing_dataframe.apply(
        lambda row: row.to_frame(), axis=1
    )
    blob.upload_from_filename.assert_called_with(TEST_FILE)

    # create a csv data source
    source = dataset.CSVDataSource(
        storage_path=f"gcs://{BUCKET}",
    )

    all_records = []
    while True:
        record = source.next_line()
        if record == None:
            break
        all_records.append(record)

    aggregated_cols = [
        list(all_records[i : i + NUMBER_OF_COLS])
        for i in range(0, len(all_records), NUMBER_OF_COLS)
    ]
    assert len(aggregated_cols) == TEST_DATASET_SIZE
    all_records_dataframe = pd.DataFrame(aggregated_cols)

    assert testing_dataframe.equals(all_records_dataframe)

    clean_up()
