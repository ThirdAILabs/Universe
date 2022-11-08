import platform
import os
import csv
import datasets
import random 
import pandas as pd 

import pytest
from thirdai import bolt

pytestmark = [pytest.mark.unit, pytest.mark.release]

TRAIN_FILE = "train_file.csv"
EVAL_FILE = "eval_file.csv"



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
    if os.path.exists(TRAIN_FILE):
        os.remove(TRAIN_FILE)

    if os.path.exists(EVAL_FILE):
        os.remove(EVAL_FILE)


def create_udt_generator_model():

    dataframe = download_grammar_correction_dataset()
    write_input_dataset_to_csv(dataframe, TRAIN_FILE)

    transformed_queries = transform_queries(dataframe=dataframe)
    write_input_dataset_to_csv(transformed_queries, EVAL_FILE)

    model = bolt.UniversalDeepTransformer(
        target_column=0, source_column=1, dataset_size="large"
    )
    model.train(filename=TRAIN_FILE)

    model.evaluate(filename=TRAIN_FILE)

def test_udt_generator_load_save():
    pass 
