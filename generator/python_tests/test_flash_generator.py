import csv
import math
import os
import random

import datasets
import pandas as pd
import pytest
from thirdai import bolt

pytestmark = [pytest.mark.unit]

QUERIES_FILE = "./queries.csv"
TRANSFORMED_QUERIES = "./transformed_queries.csv"
CONFIG_FILE = "./flash_index_config"

# The downloaded dataset from HuggingFace consists of 328 samples
DATASET_SIZE = 328


def read_csv_file(file_name):
    with open(file_name, newline="") as file:
        data = list(csv.reader(file))

    return data


def write_input_dataset_to_csv(dataframe, file_path):
    dataframe.to_csv(file_path, index=False, header=False)


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


def transform_queries(dataframe):
    """
    Randomly picks 10% of the words in the queries list and either
    removes a random character from the chosen word or applies a
    random permutation to the characters in order to create an incorrect
    version of the queries.

    """
    transformation_type = ("remove-char", "permute-string")
    transformed_dataframe = []

    for _, row in dataframe.iterrows():
        correct_query = " ".join(list(row.str.split(" ")[0]))

        incorrect_query_pair = correct_query.split(" ")
        query_length = len(incorrect_query_pair)
        words_to_permute = math.ceil(0.1 * query_length)

        index = 0
        permuted_indices = {}
        while index < words_to_permute:
            random_index = random.randint(0, words_to_permute)
            if random_index in permuted_indices:
                continue
            word_to_transform = incorrect_query_pair[random_index]

            if random.choices(transformation_type, k=1) == "remove-char":
                # Remove a random character
                char_index = random.randint(0, len(word_to_transform) - 1)
                transformed_word = (
                    word_to_transform[0:char_index:]
                    + word_to_transform[char_index + 1 : :]
                )
                incorrect_query_pair[random_index] = transformed_word

            else:
                # Permute the characters in the string
                transformed_word = "".join(
                    random.sample(word_to_transform, len(word_to_transform))
                )
                incorrect_query_pair[random_index] = transformed_word

            index += 1

        transformed_dataframe.append([correct_query, " ".join(incorrect_query_pair)])

    return pd.DataFrame(transformed_dataframe)


def delete_created_files():
    if os.path.exists(QUERIES_FILE):
        os.remove(QUERIES_FILE)
    if os.path.exists(CONFIG_FILE):
        os.remove(CONFIG_FILE)

    if os.path.exists(TRANSFORMED_QUERIES):
        os.remove(TRANSFORMED_QUERIES)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.unit
def test_flash_generator():
    """
    Tests that the generated candidate queries are reasonable given
    the input dataset.
    By default, the generator recommends top 5 closes queries from
    flash.
    """
    dataframe = download_grammar_correction_dataset()
    write_input_dataset_to_csv(dataframe, QUERIES_FILE)

    transformed_queries = transform_queries(dataframe=dataframe)
    write_input_dataset_to_csv(transformed_queries, TRANSFORMED_QUERIES)

    generator_config = bolt.GeneratorConfig(
        hash_function="DensifiedMinHash",
        num_tables=300,
        hashes_per_table=32,
        input_dim=100,
        has_incorrect_queries=True,
    )
    generator_config.save(CONFIG_FILE)

    generator = bolt.Generator(config_file_name=CONFIG_FILE)

    generator.train(file_name=TRANSFORMED_QUERIES)

    query_pairs = read_csv_file(file_name=TRANSFORMED_QUERIES)

    count_correct_results = 0
    for query_pair in query_pairs:
        generated_candidates = generator.generate(queries=[query_pair[1]])
        count_correct_results += 1 if query_pair[0] in generated_candidates[0] else 0

    assert count_correct_results / DATASET_SIZE > 0.98

    delete_created_files()
