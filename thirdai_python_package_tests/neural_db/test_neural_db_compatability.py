import os

import pandas as pd
import pytest
from thirdai import neural_db as ndb

pytestmark = [pytest.mark.release]


def current_dir():
    return os.path.dirname(os.path.abspath(__file__))


def query_file():
    return os.path.join(current_dir(), "../../auto_ml/python_tests/texts.csv")


def save_path():
    return os.path.join(current_dir(), "saved_ndbs/simple_ndb")


# This method is what was used to create the saved neural db that is located in
# the saved_ndbs directory. Since these tests are to check that old checkpoints
# still work correctly, this method is not invoked for tests, but it is included
# here for future reference.
def create_ndb():
    db = ndb.NeuralDB(embedding_dimension=512, extreme_output_dim=1000)
    db.insert([ndb.CSV(query_file(), id_column="id", weak_columns=["text"])])
    db.save(save_path())


def count_correct(db, queries, ids, retriever=None):
    correct = 0
    for query, id in zip(queries, ids):
        result = db.search(query, top_k=1, retriever=retriever)[0]
        if result.id == id:
            correct += 1
        if retriever:
            assert result.retriever == retriever

    return correct


def test_saved_ndb_accuracy():
    db = ndb.NeuralDB.from_checkpoint(save_path())

    df = pd.read_csv(query_file())

    mach_correct = count_correct(
        db=db, queries=df["text"], ids=df["id"], retriever="mach"
    )
    assert mach_correct == len(df)

    inverted_index_correct = count_correct(
        db=db, queries=df["text"], ids=df["id"], retriever="inverted_index"
    )
    assert inverted_index_correct == len(df)


def test_saved_ndb_associate():
    db = ndb.NeuralDB.from_checkpoint(save_path())

    df = pd.read_csv(query_file())

    # This is based on the tests in auto_ml/python_tests/test_rlhf.py. Basically
    # we replace each line with an acronym of that line and associate the acronyms
    # with the full text.
    df["acronym"] = df["text"].map(lambda s: "".join(w[0] for w in s.split()))

    correct_before_associate = count_correct(db=db, queries=df["acronym"], ids=df["id"])
    assert correct_before_associate < 0.1 * len(df)

    association_samples = list(zip(df["acronym"], df["text"]))
    db.associate_batch(association_samples)

    correct_after_associate = count_correct(db=db, queries=df["acronym"], ids=df["id"])
    assert correct_after_associate > 0.9 * len(df)


def test_saved_ndb_constrained_search():
    db = ndb.NeuralDB.from_checkpoint(save_path())
    # InRange "S" to "R" will match any string that starts with the letter "S".
    res = db.search("test", top_k=5, constraints={"text": ndb.InRange("S", "T")})
    assert len(res) == 5
    for r in res:
        assert r.metadata["text"][0] == "S"
