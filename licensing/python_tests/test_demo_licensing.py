import os

import pytest
import thirdai

# This line uses a hack where we can import functions from different test files
# as long as this file is run from bin/python-format.sh. To run just this file,
# run bin/python-test.sh -k "test_demo_licensing"
from model_test_utils import get_udt_census_income_model
from test_udt_simple import make_simple_trained_model

pytestmark = [pytest.mark.release]

# I created this key on Keygen, it should be good only for the small census
# dataset (the first 10 lines of the normal census training set)
SMALL_CENSUS_KEY = "HKRW-4R97-KRLY-TMVY-79TW-3FHH-NRHR-HLYC"


def test_census_key_works_on_small_census():
    thirdai.licensing.activate(SMALL_CENSUS_KEY)
    train_filename, _, _ = thirdai.demos.download_census_income()
    small_census_filename = "small_census.txt"
    with open(small_census_filename, "w") as output:
        with open(train_filename) as input:
            for i, line in enumerate(input):
                if i > 10:
                    break
                output.write(f"{line}")
    model = get_udt_census_income_model()
    model.train(small_census_filename, epochs=1, learning_rate=0.01)
    os.remove(small_census_filename)


def test_census_key_fails_on_others():
    thirdai.licensing.activate(SMALL_CENSUS_KEY)
    with pytest.raises(
        RuntimeError,
        match="This dataset is not authorized under this license.",
    ):
        make_simple_trained_model()


# This fixture removes the stored access key after each test finishes, ensuring
# that other tests that run in this pytest environment will get a clean
# licensing slate
@pytest.fixture(autouse=True)
def set_license_back_to_valid():
    # The yield means that pytest will wait until the test finishes to run
    # the code below it
    yield
    thirdai.licensing.deactivate()
