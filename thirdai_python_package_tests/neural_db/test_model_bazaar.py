import os
import shutil

import pytest
from thirdai.neural_db import Bazaar

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
