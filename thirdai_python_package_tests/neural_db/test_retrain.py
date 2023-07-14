import os

import pytest
from thirdai import neural_db

pytestmark = [pytest.mark.unit, pytest.mark.release]


@pytest.fixture
def create_simple_dataset():
    filename = "simple.csv"
    with open(filename, "w") as file:
        file.writelines(["text,id\n", "apples are red,0\n", "spinach is green,1\n"])

    yield filename

    os.remove(filename)


def test_neural_db_associate(create_simple_dataset):
    filename = create_simple_dataset
    ndb = neural_db.NeuralDB("")

    doc = neural_db.CSV(
        filename,
        id_column="id",
        strong_columns=["text"],
        weak_columns=["text"],
        reference_columns=[],
    )

    ndb.insert(sources=[doc])

    ndb.associate_batch([("fruit", "apple"), ("vegetable", "spinach")])

    ndb.retrain()
