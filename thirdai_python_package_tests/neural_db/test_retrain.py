import pytest
from ndb_utils import create_simple_dataset
from thirdai import neural_db

pytestmark = [pytest.mark.unit, pytest.mark.release]


def test_neural_db_retrain(create_simple_dataset):
    filename = create_simple_dataset
    ndb = neural_db.NeuralDB("")

    doc = neural_db.CSV(
        filename,
        id_column="label",
        strong_columns=["text"],
        weak_columns=["text"],
        reference_columns=[],
    )

    ndb.insert(sources=[doc])

    ndb.associate_batch([("fruit", "apple"), ("vegetable", "spinach")])

    ndb.retrain()
