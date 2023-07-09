import pytest
import time
import os
import shutil
from pathlib import Path


@pytest.mark.unit
def test_model_bazaar():
    from thirdai import neural_db as ndb

    cache = Path("./bazaar_cache")
    # Throws an error if path exists, so we know this is a clean folder if it
    # does not fail.
    os.mkdir(cache)

    bazaar = ndb.Bazaar(cache)
    bazaar.fetch()
    bazaar_models = bazaar.list_model_names()

    # TODO(Geordie): We should have a test model that is smaller, guaranteed
    # to be there (will not be cleared permanently), and possibly invisible to
    # other users.
    model_name = "Q&A V2"
    assert model_name in bazaar_models
    checkpoint = bazaar.get_model_dir("Q&A V2")

    # No assertion since we only need to know that it does not break.
    model = ndb.NeuralDB("user")
    model.from_checkpoint(checkpoint)

    # We will load the same model again to check whether we successfully cached
    # it. The assertion is based on elapsed time, which is typically avoided
    # since they can be flakey, but this should finish in far less than the
    # assertion threshold of 5 seconds, and 5 seconds is much faster than what
    # it takes to download the model.
    start = time.time()
    checkpoint = bazaar.get_model_dir("Q&A V2")
    end = time.time()
    elapsed = end - start
    assert elapsed < 5

    shutil.rmtree(checkpoint)
