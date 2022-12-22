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


@pytest.fixture(scope="session")
def download_clinc_dataset():
    return download_clinc_dataset_wrapped()


@pytest.fixture(scope="session")
def download_internet_ads_dataset():
    return download_internet_ads_dataset_wrapped()


@pytest.fixture(scope="session")
def download_census_income():
    return download_census_income_wrapped(
        num_inference_samples="all", return_labels=True
    )


@pytest.fixture(scope="session")
def download_brazilian_houses_dataset():
    return download_brazilian_houses_dataset_wrapped()
