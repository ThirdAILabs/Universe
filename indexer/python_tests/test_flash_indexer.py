import os
import pytest
from thirdai import bolt, deployment
import datasets
import pandas as pd

pytestmark = [pytest.mark.integration, pytest.mark.release]

QUERIES_FILE = "./queries.csv"
CONFIG_FILE = "./flash_index_config"

@pytest.fixture
def download_grammar_correction_dataset():
    dataset = datasets.load_dataset("snips_built_in_intents", "small")
    dataframe = pd.DataFrame(dataset)
    extracted_text = []
    for _, row in dataframe.iterrows():
        extracted_text.append(row.to_dict()['train']['text'])
    
    return pd.DataFrame(extracted_text)


@pytest.fixture
def write_input_dataset_to_csv(dataframe, file_path):
    dataframe.to_csv(file_path, index=False, header=False)

@pytest.fixture
def delete_created_files():
    if os.path.exists(QUERIES_FILE):
        os.remove(QUERIES_FILE)

@pytest.mark.unit
def flash_indexer_test():
    query_dataframe = download_grammar_correction_dataset()
    write_input_dataset_to_csv(query_dataframe, QUERIES_FILE)

    indexer_config = deployment.FlashIndexConfig(
        hash_function="DensifiedMinHash",
        num_tables=100,
        hashes_per_table=15,
        input_dim=100
    )
    indexer_config.save(CONFIG_FILE)

    indexer = deployment.Indexer(
        config_file_name=CONFIG_FILE
    )
    
    generator = indexer.build_index(file_name=QUERIES_FILE)

    output = generator.generate(query="share my curnt locatio")

    delete_created_files()





if __name__ == "__main__":
    text = download_grammar_correction_dataset()
    write_input_dataset_to_csv(text, QUERIES_FILE)
