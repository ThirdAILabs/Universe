import os

import pytest
import thirdai

# These lines use a hack where we can import functions from different test files
# as long as this file is run from bin/python-test.sh. To run just this file,
# run bin/python-test.sh -k "test_demo_licensing"
from model_test_utils import get_udt_census_income_model
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
    model.train(small_census_filename, epochs=3, learning_rate=0.01)
    model.save("small_census_model.serialized")
    model = bolt.UniversalDeepTransformer.load("small_census_model.serialized")


def test_census_demo_key_fails_on_udt():
    thirdai.licensing.activate(SMALL_CENSUS_KEY)
    with pytest.raises(
        RuntimeError,
        match="This dataset is not authorized under this license.",
    ):
        make_simple_trained_model()


def test_census_demo_key_fails_on_query_reformulation():
    thirdai.licensing.activate(SMALL_CENSUS_KEY)

    temp_filename = "temp_query_reformulation.txt"
    with open(temp_filename, "w") as file:
        file.writelines(["query,label\n", "input1,output1\n", "input2,output2\n"])

    model = bolt.UniversalDeepTransformer(
        source_column="query",
        target_column="label",
        dataset_size="small",
    )

    with pytest.raises(
        RuntimeError,
        match="This dataset is not authorized under this license.",
    ):
        model.train(temp_filename)

    os.remove(temp_filename)
