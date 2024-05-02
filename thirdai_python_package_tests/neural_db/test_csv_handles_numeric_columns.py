import os
import uuid

import pandas as pd
import pytest
from thirdai import neural_db as ndb

pytestmark = [pytest.mark.unit]


@pytest.fixture(scope="session")
def csv():
    path = f"numeric_csv_{uuid.uuid4()}.csv"
    pd.DataFrame(
        {
            "col": [0, 1, 2, 3, 4],
        }
    ).to_csv(path, index=None)
    yield path
    os.remove(path)


def test_csv_handles_numeric_columns(csv):
    # Just make sure it doesn't break, which it did previously.
    db = ndb.NeuralDB()
    db.insert(ndb.CSV(csv, strong_columns=["col"], weak_columns=["col"]))
    db.search("0", top_k=1)
