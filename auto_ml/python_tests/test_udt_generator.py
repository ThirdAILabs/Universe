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


def read_csv_file(file_name: str) -> List[List[str]]:
    with open(file_name, newline="") as file:
        data = list(csv.reader(file))

    return data


def write_input_dataset_to_csv(dataframe: pd.DataFrame, file_path: str) -> None:
    dataframe.to_csv(file_path, index=False, header=False)


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


def train_udt_query_reformulation_model() -> bolt.UniversalDeepTransformer:
    model = bolt.UniversalDeepTransformer(
        target_column_index=0, source_column_index=1, dataset_size="small"
    )
    model.train(filename=TRAIN_FILE_PATH)
    return model


@pytest.mark.filterwarnings("ignore")
def test_udt_generator_load_save(prepared_datasets):
    model = train_udt_query_reformulation_model()
    model.save(MODEL_PATH)

    deserialized_model = bolt.UniversalDeepTransformer.load(
        MODEL_PATH, model_type="udt_generator"
    )

    eval_outputs = model.evaluate(filename=TRAIN_FILE_PATH, top_k=5)
    print(f"type of model = {type(deserialized_model)}")
    deserialized_model_eval_outputs = deserialized_model.evaluate(
        filename=TRAIN_FILE_PATH, top_k=5
    )
    for index in range(len(eval_outputs)):
        assert eval_outputs[index] == deserialized_model_eval_outputs[index]


@pytest.mark.filterwarnings("ignore")
def test_udt_generator_raises_attribute_error(prepared_datasets):
    """
    This test ensures that certain methods are only available for
    specific UDT models. For instance, a user should not be able to
    call model.embedding_representation() when the underlying model
    is an instance of a query-reformulation-specific UDT.
    """

    single_test_sample = {
        "userId": "0",
        "movieId": "1",
        "timestamp": "2022-08-31",
        "hoursWatched": "1",
    }
    model = train_udt_query_reformulation_model()
    with pytest.raises(
        AttributeError,
    ):
        model.embedding_representation(input_sample=single_test_sample)

    with pytest.raises(
        AttributeError,
    ):
        model.class_name(neuron_id=5)

    with pytest.raises(
        AttributeError,
    ):
        model.index(input_sample=single_test_sample)

    with pytest.raises(
        AttributeError,
    ):
        model.index_batch(input_samples=[single_test_sample] * 5)

    with pytest.raises(
        AttributeError,
    ):
        model.reset_temporal_trackers()

    with pytest.raises(
        AttributeError,
    ):
        model.explain(input_sample=single_test_sample)

    delete_created_files()
