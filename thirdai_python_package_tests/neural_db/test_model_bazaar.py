import os
import shutil

import pytest
from thirdai.neural_db import Bazaar
from thirdai.neural_db.model_bazaar.bazaar import BazaarEntry

pytestmark = [pytest.mark.unit]


BAZAAR_CACHE = "./test_cache"


def clear_cache():
    if os.path.exists(BAZAAR_CACHE):
        shutil.rmtree(BAZAAR_CACHE)


def test_model_bazaar_registry_cache():
    clear_cache()
    bazaar_1 = Bazaar(cache_dir=BAZAAR_CACHE)
    bazaar_1.fetch()
    assert len(bazaar_1._registry) > 0
    bazaar_2 = Bazaar(cache_dir=BAZAAR_CACHE)
    for key, val in bazaar_1._registry.items():
        assert val == bazaar_2._registry[key]


def test_model_bazaar_fetch_remove_outdated_flag():
    clear_cache()
    bazaar = Bazaar(cache_dir=BAZAAR_CACHE)
    bazaar._registry["fake_entry"] = BazaarEntry(
        display_name="fake_model",
        trained_on="fake_data",
        num_params="fake_num_params",
        size=1000,
        hash="fake_hash",
    )
    bazaar.fetch(remove_outdated=False)
    assert bazaar._registry["fake_entry"].display_name == "fake_model"
    assert bazaar._registry["fake_entry"].trained_on == "fake_data"
    assert bazaar._registry["fake_entry"].num_params == "fake_num_params"
    assert bazaar._registry["fake_entry"].size == 1000
    assert bazaar._registry["fake_entry"].hash == "fake_hash"
    bazaar.fetch()
    assert not "fake_entry" in bazaar._registry
