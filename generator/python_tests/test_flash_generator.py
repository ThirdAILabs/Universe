import os

import datasets
import pandas as pd
import pytest
from thirdai import bolt

pytestmark = [pytest.mark.unit]

QUERIES_FILE = "./queries.csv"
CONFIG_FILE = "./flash_index_config"


def download_grammar_correction_dataset():
    """
    The grammar correction dataset is retrieved from HuggingFace:
    https://huggingface.co/datasets/snips_built_in_intents
    """
    dataset = datasets.load_dataset("snips_built_in_intents", "small")
    dataframe = pd.DataFrame(dataset)
    extracted_text = []
    for _, row in dataframe.iterrows():
        extracted_text.append(row.to_dict()["train"]["text"])

    return pd.DataFrame(extracted_text)


def write_input_dataset_to_csv(dataframe, file_path):
    dataframe.to_csv(file_path, index=False, header=False)


def delete_created_files():
    if os.path.exists(QUERIES_FILE):
        os.remove(QUERIES_FILE)
    if os.path.exists(CONFIG_FILE):
        os.remove(CONFIG_FILE)


def get_queries_for_testing():
    queries_with_candidate_recommendations = {
        "Send my location": {
            "Send my location",
            "Send my location to my husband",
            "Send my location to Anna",
            "Send my current location",
        },
        "Shre my eta wth mry": {
            "Share my ETA with Mary Jane",
            "Share my ETA with Jo",
            "Share my arrival time with Juan Carlos",
        },
    }
    return queries_with_candidate_recommendations


@pytest.mark.filterwarnings("ignore")
@pytest.mark.unit
def test_flash_generator():
    """
    Tests that the generated candidate queries are reasonable given
    the input dataset.
    By default, the generator recommends top 5 closes queries from
    flash. This unit test checks that at least two of the expected
    queries are present in the 5 generated queries by the generator.
    """
    if not os.path.exists(QUERIES_FILE):
        query_dataframe = download_grammar_correction_dataset()
        write_input_dataset_to_csv(query_dataframe, QUERIES_FILE)

    generator_config = bolt.GeneratorConfig(
        hash_function="DensifiedMinHash",
        num_tables=300,
        hashes_per_table=32,
        input_dim=100,
    )
    generator_config.save(CONFIG_FILE)

    generator = bolt.Generator(config_file_name=CONFIG_FILE)

    generator.train(file_name=QUERIES_FILE, has_incorrect_queries=False)

    queries_with_expected_outputs = get_queries_for_testing()

    for query in queries_with_expected_outputs:
        generated_candidate_queries = generator.generate(queries=[query])
        expected_outputs = queries_with_expected_outputs[query]

        assert (
            len(set(generated_candidate_queries[0]).intersection(expected_outputs)) >= 2
        )

    delete_created_files()
