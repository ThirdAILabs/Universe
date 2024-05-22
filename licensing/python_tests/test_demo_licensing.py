import os

import pytest
import thirdai
from download_dataset_fixtures import download_clinc_dataset
from licensing_utils import deactivate_license_at_start_of_demo_test

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
CLINC_DATASET_KEY = "PRCH-AAVV-3ART-3WY9-PM9H-KULT-PRW3-C399"


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


def test_census_demo_keys_fails_on_inverted_index():
    thirdai.licensing.activate(SMALL_CENSUS_KEY)
    index = thirdai.search.InvertedIndex()
    with pytest.raises(
        RuntimeError,
        match="This dataset is not authorized under this license.",
    ):
        index.index([1, 2], ["a", "b"])


def test_census_demo_key_fails_on_query_reformulation():
    thirdai.licensing.activate(SMALL_CENSUS_KEY)

    temp_filename = "temp_query_reformulation.txt"
    with open(temp_filename, "w") as file:
        file.writelines(["query,label\n", "input1,output1\n", "input2,output2\n"])

    model = bolt.UniversalDeepTransformer(
        data_types={
            "query": bolt.types.text(),
            "label": bolt.types.text(),
        },
        target="label",
        options={"dataset_size": "small"},
    )

    with pytest.raises(
        RuntimeError,
        match="This dataset is not authorized under this license.",
    ):
        model.train(temp_filename)

    os.remove(temp_filename)


def simple_mach_model(target_column="label"):
    return bolt.UniversalDeepTransformer(
        data_types={
            "query": bolt.types.text(),
            target_column: bolt.types.categorical(n_classes=10, type="int"),
        },
        target=target_column,
        options={"extreme_classification": True, "extreme_output_dim": 100},
    )


def test_introduce_document_fails_on_demo_license():
    thirdai.licensing.activate(SMALL_CENSUS_KEY)

    temp_filename = "temp_data.txt"
    with open(temp_filename, "w") as file:
        file.writelines(["query,label\n", "input1,1\n", "input2,2\n"])

    model = simple_mach_model()

    model.clear_index()

    with pytest.raises(
        RuntimeError,
        match="This dataset is not authorized under this license.",
    ):
        model.introduce_documents(
            temp_filename, strong_column_names=[], weak_column_names=["query"]
        )

    with pytest.raises(
        RuntimeError,
        match="The license was found to be invalid: You must have a full license to perform this operation.",
    ):
        model.introduce_document(
            {"text": "some text"},
            strong_column_names=[],
            weak_column_names=["text"],
            label=1000,
        )

    with pytest.raises(
        RuntimeError,
        match="The license was found to be invalid: You must have a full license to perform this operation.",
    ):
        model.introduce_label(
            [{"text": "some text"}],
            label=1000,
        )


def test_introduce_documents_works_on_clinc(download_clinc_dataset):
    thirdai.licensing.activate(CLINC_DATASET_KEY)

    train_data, _, _ = download_clinc_dataset

    model = simple_mach_model(target_column="category")
    model.clear_index()

    model.introduce_documents(
        train_data, strong_column_names=[], weak_column_names=["text"]
    )


def test_set_index_fails_on_demo_license():
    thirdai.licensing.activate(SMALL_CENSUS_KEY)

    model = simple_mach_model()

    model.get_index()  # This should work since it's used in neural_db

    with pytest.raises(
        RuntimeError,
        match="The license was found to be invalid: You must have a full license to perform this operation.",
    ):
        model.set_index(None)


def test_upvote_fails_on_demo_license():
    thirdai.licensing.activate(SMALL_CENSUS_KEY)

    model = simple_mach_model()

    with pytest.raises(
        RuntimeError,
        match="The license was found to be invalid: You must have a full license to perform this operation.",
    ):
        model.upvote([])
