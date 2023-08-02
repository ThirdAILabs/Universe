import os

import pytest
from thirdai import neural_db


@pytest.fixture
def create_simple_dataset():
    filename = "simple.csv"
    with open(filename, "w") as file:
        file.writelines(
            [
                "text,id\n",
                "apples are red,0\n",
                "spinach is green,1\n",
                "bananas are yellow,2\n",
                "oranges are orange,3\n",
            ]
        )

    yield filename

    os.remove(filename)


@pytest.fixture
def train_simple_neural_db(create_simple_dataset):
    filename = create_simple_dataset
    ndb = neural_db.NeuralDB()

    doc = neural_db.CSV(
        filename,
        id_column="id",
        strong_columns=["text"],
        weak_columns=["text"],
        reference_columns=["id", "text"],
    )

    ndb.insert(sources=[doc], train=True)

    return ndb
