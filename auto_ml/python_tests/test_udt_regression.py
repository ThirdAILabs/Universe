import pytest
from thirdai import bolt
from download_datasets import download_brazilian_houses_dataset


pytestmark = [pytest.mark.unit, pytest.mark.release]



@pytest.fixture(scope="module")
def train_udt_regression(download_brazilian_houses_dataset):
    train_filename, _, _ = download_brazilian_houses_dataset
    model = bolt.UniversalDeepTransformer(
        data_types={
            "area": bolt.types.numerical(range=(11, 46350)),
            "rooms": bolt.types.categorical(),
            "bathroom": bolt.types.categorical(),
            "parking_spaces": bolt.types.categorical(),
            "hoa_(BRL)": bolt.types.numerical(range=(0, 1117000)),
            "rent_amount_(BRL)": bolt.types.numerical(range=(450, 45000)),
            "property_tax_(BRL)": bolt.types.numerical(range=(0, 313700)),
            "fire_insurance_(BRL)": bolt.types.numerical(range=(3, 677)),
            "totalBRL": bolt.types.numerical(range=(6, 14)),
        },
        target="totalBRL",
    )

    train_config = bolt.TrainConfig(epochs=5, learning_rate=0.01)
    model.train(train_filename, train_config)



def test_udt_regression_accuracy(train_udt_regression):
    pass
