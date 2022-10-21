import os

import datasets
import pandas as pd
import pytest
from thirdai import bolt

pytestmark = [pytest.mark.unit]

QUERIES_FILE = "./queries.csv"
CONFIG_FILE = "./flash_index_config"


def download_grammar_correction_dataset():
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


@pytest.mark.filterwarnings("ignore")
@pytest.mark.unit
def test_flash_generator():
    query_dataframe = download_grammar_correction_dataset()
    write_input_dataset_to_csv(query_dataframe, QUERIES_FILE)

    generator_config = bolt.GeneratorConfig(
        hash_function="FastSRP",
        num_tables=100,
        hashes_per_table=15,
        input_dim=100,
    )
    generator_config.save(CONFIG_FILE)

    generator = bolt.Generator(config_file_name=CONFIG_FILE)

    generator.train_(file_name=QUERIES_FILE)

    candidate_queries = generator.generate(
        queries=["Share my location with my Uber driver"]
    )

    print(f"CANDIDATE QUERIES = {candidate_queries}")

    delete_created_files()
