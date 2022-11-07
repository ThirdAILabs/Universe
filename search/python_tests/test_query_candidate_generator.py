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

    The input is expected to be a Pandas DataFrame with one column
    containing the correct queries. The output is expected to be
    another Pandas DataFrame with two columns. The first column remains
    the same, but the second column consists of queries transformed
    according to the rule detailed above.

    """
    transformation_type = ("remove-char", "permute-string")
    transformed_dataframe = []

    for _, row in dataframe.iterrows():
        correct_query = " ".join(list(row.str.split(" ")[0]))

        incorrect_query_pair = correct_query.split(" ")
        query_length = len(incorrect_query_pair)
        words_to_transform = math.ceil(0.1 * query_length)

        transformed_words = 0
        visited_indices = set()

        while transformed_words < words_to_transform:
            random_index = random.randint(0, words_to_transform)
            if random_index in visited_indices:
                continue
            word_to_transform = incorrect_query_pair[random_index]

            if random.choices(transformation_type, k=1) == "remove-char":
                # Remove a random character
                char_index = random.randint(0, len(word_to_transform) - 1)
                transformed_word = (
                    word_to_transform[0:char_index]
                    + word_to_transform[char_index + 1 :]
                )
                incorrect_query_pair[random_index] = transformed_word

            else:
                # Permute the characters in the string
                transformed_word_char_list = list(word_to_transform)
                random.shuffle(transformed_word_char_list)

                incorrect_query_pair[random_index] = "".join(transformed_word_char_list)

            visited_indices.add(random_index)
            transformed_words += 1

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

    generator_config = bolt.models.GeneratorConfig(
        hash_function="MinHash",
        num_tables=20,
        hashes_per_table=10,
        range=100,
        n_grams=[3, 4],
        has_incorrect_queries=True,
    )
    generator_config.save(CONFIG_FILE)

    generator = bolt.models.Generator(config_file_name=CONFIG_FILE)
    generator.train(file_name=TRANSFORMED_QUERIES)

    query_pairs = read_csv_file(file_name=TRANSFORMED_QUERIES)

    generated_candidates = generator.generate(
        queries=[query_pair[1] for query_pair in query_pairs], top_k=5
    )

    correct_results = 0
    for query_index in range(len(query_pairs)):
        correct_results += (
            1 if query_pairs[query_index][0] in generated_candidates[query_index] else 0
        )

    recall = correct_results / len(query_pairs)
    assert recall > 0.95

    delete_created_files()
