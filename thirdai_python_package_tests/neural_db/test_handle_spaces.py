from test_neural_db_compatability import save_path
from thirdai import neural_db as ndb


def test_handle_spaces_in_csv_column():
    # write simple csv, columns have spaces
    # get references, assert that ref metadata keys doesn't have spaces

    pass


def test_handle_spaces_in_metadata():
    pass


def test_handle_spaces_backwards_compatibility():
    db = ndb.NeuralDB.from_checkpoint(save_path())
    # db.
    pass
