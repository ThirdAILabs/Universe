import os

import numpy as np
import pytest
import thirdai

# These lines use a hack where we can import functions from different test files
# as long as this file is run from bin/python-format.sh. To run just this file,
# run bin/python-test.sh -k "test_demo_licensing"
from model_test_utils import get_udt_census_income_model
from test_udt_generator import TRAIN_SOURCE_TARGET_FILE as udt_query_reformulation_file
from test_udt_generator import grammar_correction_dataset
from test_udt_generator import prepared_datasets as udt_query_reformulation_fixture
from test_udt_generator import train_udt_query_reformulation_model
from test_udt_simple import make_simple_trained_model
from thirdai import bolt

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


def test_census_demo_key_fails_save_load():
    thirdai.licensing.activate(SMALL_CENSUS_KEY)
    model = get_udt_census_income_model()
    with pytest.raises(
        RuntimeError, match=".*You must have a full license to save and load models.*"
    ):
        model.save("test")


def test_census_demo_key_fails_with_bolt_api():
    thirdai.licensing.activate(SMALL_CENSUS_KEY)
    input = bolt.nn.Input(dim=1)
    output = bolt.nn.FullyConnected(dim=1, activation="relu")(input)
    model = bolt.nn.Model(inputs=[input], output=output)
    model.compile(bolt.nn.losses.MeanSquaredError())
    input_data = thirdai.dataset.from_numpy(
        data=np.array([[1.0]], dtype="float32"), batch_size=1
    )
    output_data = thirdai.dataset.from_numpy(
        data=np.array([[1.0]], dtype="float32"), batch_size=1
    )
    train_config = bolt.TrainConfig(epochs=1, learning_rate=0.1)
    with pytest.raises(
        RuntimeError, match="You must have a full license to perform this operation"
    ):
        model.train(input_data, output_data, train_config)


def test_census_demo_key_fails_on_udt():
    thirdai.licensing.activate(SMALL_CENSUS_KEY)
    with pytest.raises(
        RuntimeError,
        match="This dataset is not authorized under this license.",
    ):
        make_simple_trained_model()


def test_census_demo_key_fails_on_generator(udt_query_reformulation_fixture):
    thirdai.licensing.activate(SMALL_CENSUS_KEY)
    with pytest.raises(
        RuntimeError,
        match="This dataset is not authorized under this license.",
    ):
        train_udt_query_reformulation_model(udt_query_reformulation_file)


# This fixture removes the stored access key after each test finishes, ensuring
# that other tests that run in this pytest environment will get a clean
# licensing slate
@pytest.fixture(autouse=True)
def set_license_back_to_valid():
    # The yield means that pytest will wait until the test finishes to run
    # the code below it
    yield
    thirdai.licensing.deactivate()
