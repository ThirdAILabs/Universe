import csv
import math
import os
import random
from typing import List

import datasets
import pandas as pd
import pytest
from thirdai import bolt

pytestmark = [pytest.mark.unit, pytest.mark.release]

TRAIN_FILE_PATH = "./query_reformulation.csv"
MODEL_PATH = "udt_generator_model.bolt"

RECALL_THRESHOLD = 0.95


def read_csv_file(file_name: str) -> List[List[str]]:
    with open(file_name, newline="") as file:
        data = list(csv.reader(file))

    # Remove the file header
    data = data[1:]
    return data


def write_input_dataset_to_csv(dataframe: pd.DataFrame, file_path: str) -> None:
    # Add file header since the "train" and "evaluate" methods assume the
    # input CSV file has a header.
    dataframe.columns = ["target_column", "source_column"]

    # TODO(Geordie): Fix this when the new CSV parser is in
    dataframe = dataframe.replace(",", "", regex=True)

    dataframe.to_csv(file_path, index=False)


@pytest.fixture(scope="module")
def grammar_correction_dataset() -> pd.DataFrame:
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


def transform_queries(dataframe: pd.DataFrame) -> pd.DataFrame:
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


@pytest.fixture
def prepared_datasets(grammar_correction_dataset) -> None:
    transformed_queries = transform_queries(dataframe=grammar_correction_dataset)
    write_input_dataset_to_csv(transformed_queries, TRAIN_FILE_PATH)


def delete_created_files() -> None:
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)

    if os.path.exists(TRAIN_FILE_PATH):
        os.remove(TRAIN_FILE_PATH)


def run_generator_test(
    model: bolt.models.UDTGenerator, source_col_index: int, target_col_index: int
) -> None:
    """
    Tests that the generated candidate queries are reasonable given
    the input dataset.
    By default, the generator recommends top 5 closes queries from
    flash.
    """

    query_pairs = read_csv_file(file_name=TRAIN_FILE_PATH)

    queries = [query_pair[source_col_index] for query_pair in query_pairs]
    generated_candidates = model.predict_batch(queries=queries, top_k=5)

    correct_results = 0
    for query_index in range(len(query_pairs)):
        correct_results += (
            1
            if query_pairs[query_index][target_col_index]
            in generated_candidates[query_index]
            else 0
        )

    recall = correct_results / len(query_pairs)
    assert recall > RECALL_THRESHOLD


def train_udt_query_reformulation_model() -> bolt.UniversalDeepTransformer:
    model = bolt.UniversalDeepTransformer(
        source_column="source_column",
        target_column="target_column",
        dataset_size="small",
    )
    model.train(filename=TRAIN_FILE_PATH)
    return model


@pytest.mark.filterwarnings("ignore")
def test_udt_generator(prepared_datasets):
    model = train_udt_query_reformulation_model()
    run_generator_test(model=model, source_col_index=1, target_col_index=0)


@pytest.mark.filterwarnings("ignore")
def test_udt_generator_load_and_save(prepared_datasets):
    trained_model = train_udt_query_reformulation_model()
    run_generator_test(model=trained_model, source_col_index=1, target_col_index=0)

    trained_model.save(filename=MODEL_PATH)

    deserialized_model = bolt.UniversalDeepTransformer.load(filename=MODEL_PATH)
    model_eval_outputs = trained_model.evaluate(filename=TRAIN_FILE_PATH, top_k=5)
    deserialized_model_outputs = deserialized_model.evaluate(
        filename=TRAIN_FILE_PATH, top_k=5
    )

    for index in range(len(model_eval_outputs)):
        assert model_eval_outputs[index] == deserialized_model_outputs[index]

    delete_created_files()
