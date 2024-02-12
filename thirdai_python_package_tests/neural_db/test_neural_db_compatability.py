import os
from thirdai import neural_db as ndb
import pandas as pd


def current_dir():
    return os.path.dirname(os.path.abspath(__file__))


def query_file():
    return os.path.join(current_dir(), "../../auto_ml/python_tests/texts.csv")


def save_path():
    return os.path.join(current_dir(), "saved_ndbs/simple_ndb")


# This method is what was used to create the saved neural db that is located in
# the saved_ndbs directory. Since these tests are to check that old checkpoints
# still work correctly, this method is not invoked for tests, but it is included
# for future reference.
def create_ndb():
    db = ndb.NeuralDB()
    db.insert(ndb.CSV(query_file(), id_column="id", weak_columns=["text"]))
    db.save(save_path())


def test_saved_ndb_accuracy():
    db = ndb.NeuralDB.from_checkpoint(save_path())

    df = pd.read_csv()

    correct = 0
    for row in df.iterrows():
        result = db.search(row["text"], top_k=1)[0]

        if result.id == row["id"]:
            correct += 1

    assert correct / len(df) >= 0.9
