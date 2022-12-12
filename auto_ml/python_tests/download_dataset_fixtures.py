import os

import pytest
from thirdai.demos import (
    download_brazilian_houses_dataset as download_brazilian_houses_dataset_wrapped,
)
from thirdai.demos import download_census_income as download_census_income_wrapped
from thirdai.demos import download_clinc_dataset as download_clinc_dataset_wrapped
from thirdai.demos import (
    download_internet_ads_dataset as download_internet_ads_dataset_wrapped,
)


def wrap(download_func):
    train, test, inference = download_func()
    yield train, test, inference
    os.remove(train)
    os.remove(test)


@pytest.fixture(scope="session")
def download_clinc_dataset():
    yield from wrap(download_clinc_dataset_wrapped)


@pytest.fixture(scope="session")
def download_internet_ads_dataset():
    yield from wrap(download_internet_ads_dataset_wrapped)


@pytest.fixture(scope="session")
def download_census_income():
    yield from wrap(
        lambda: download_census_income_wrapped(
            num_inference_samples="all", return_labels=True
        )
    )


@pytest.fixture(scope="session")
def download_brazilian_houses_dataset():
    yield from wrap(download_brazilian_houses_dataset_wrapped)
