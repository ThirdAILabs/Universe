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
    db = ndb.NeuralDB(embedding_dimension=512, extreme_output_dim=1000, fhr=25000)
    db.insert([ndb.CSV(query_file(), id_column="id", weak_columns=["text"])])
    db.save(save_path())


create_ndb()
