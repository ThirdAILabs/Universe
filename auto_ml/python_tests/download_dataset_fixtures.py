import pytest
from thirdai.demos import (
    download_amazon_kaggle_product_catalog_sampled as download_amazon_kaggle_product_catalog_sampled_wrapped,
)
from thirdai.demos import download_beir_dataset
from thirdai.demos import (
    download_brazilian_houses_dataset as download_brazilian_houses_dataset_wrapped,
)
from thirdai.demos import download_census_income as download_census_income_wrapped
from thirdai.demos import download_clinc_dataset as download_clinc_dataset_wrapped
from thirdai.demos import (
    download_internet_ads_dataset as download_internet_ads_dataset_wrapped,
)
from thirdai.demos import download_mnist_dataset as download_mnist_dataset_wrapped
from thirdai.demos import download_yelp_chi_dataset as download_yelp_chi_dataset_wrapped


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


@pytest.fixture(scope="session")
def download_mnist_dataset():
    return download_mnist_dataset_wrapped()


@pytest.fixture(scope="session")
def download_yelp_chi_dataset():
    return download_yelp_chi_dataset_wrapped()


@pytest.fixture(scope="session")
def download_amazon_kaggle_product_catalog_sampled():
    return download_amazon_kaggle_product_catalog_sampled_wrapped()


@pytest.fixture(scope="session")
def download_scifact_dataset():
    return download_beir_dataset("scifact")
