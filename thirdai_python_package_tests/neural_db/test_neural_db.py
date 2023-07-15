
from ndb_utils import create_simple_dataset
import pytest
from thirdai import neural_db

pytestmark = [pytest.mark.unit, pytest.mark.release]

def test_neural_db_save_load(create_simple_dataset):
    filename = create_simple_dataset
    ndb = neural_db.NeuralDB()

    doc = neural_db.CSV(
        filename,
        id_column="id",
        strong_columns=["text"],
        weak_columns=["text"],
        reference_columns=[],
    )

    ndb.insert(sources=[doc], train=True)

    ndb.save("temp")




