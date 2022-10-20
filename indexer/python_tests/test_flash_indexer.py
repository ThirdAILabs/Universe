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
def test_flash_indexer():
    query_dataframe = download_grammar_correction_dataset()
    write_input_dataset_to_csv(query_dataframe, QUERIES_FILE)

    indexer_config = bolt.IndexerConfig(
        hash_function="FastSRP",
        num_tables=100,
        hashes_per_table=15,
        input_dim=100,
    )
    indexer_config.save(CONFIG_FILE)

    indexer = bolt.Indexer(config_file_name=CONFIG_FILE)

    generator = indexer.build_index(file_name=QUERIES_FILE)

    candidate_queries = generator.generate(
        query="Share my location with my Uber driver"
    )

    delete_created_files()
