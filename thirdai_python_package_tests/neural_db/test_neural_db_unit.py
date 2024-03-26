import pytest
from ndb_utils import create_simple_dataset
from thirdai import neural_db as ndb
from thirdai.neural_db.models.mach_defaults import autotune_from_scratch_min_max_epochs

pytestmark = [pytest.mark.unit]


def test_custom_epoch(create_simple_dataset):
    db = ndb.NeuralDB(user_id="user")

    doc = ndb.CSV(
        path=create_simple_dataset,
        id_column="label",
        strong_columns=["text"],
        weak_columns=["text"],
        reference_columns=["text"],
    )

    num_epochs = 10
    db.insert(sources=[doc], epochs=num_epochs)

    assert num_epochs == db._savable_state.model.get_model()._get_model().epochs()


def test_neuraldb_stopping_condition(create_simple_dataset):
    db = ndb.NeuralDB(user_id="user")

    doc = ndb.CSV(
        path=create_simple_dataset,
        id_column="label",
        strong_columns=["text"],
        weak_columns=["text"],
        reference_columns=["text"],
    )

    db.insert(sources=[doc])

    min_epochs, _ = autotune_from_scratch_min_max_epochs(size=1)

    # Our training stops when epochs >= min_epochs and accuracy >= 0.95
    # Since there is only 1 sample in the CSV the db should stop at min_epochs
    model_epoch_count = db._savable_state.model.get_model()._get_model().epochs()
    assert min_epochs == model_epoch_count - 1
