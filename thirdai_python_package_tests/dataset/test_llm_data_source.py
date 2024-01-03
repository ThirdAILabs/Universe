import json

import pytest
from thirdai import bolt, dataset
from thirdai.dataset import LLMDataSource, UnifiedLLMDataSource


@pytest.mark.unit
def test_llm_data_source():
    VOCAB_SIZE = 50267

    featurizer = dataset.TextGenerationFeaturizer(
        lrc_len=3, irc_len=2, src_len=1, vocab_size=VOCAB_SIZE
    )
    # Sample JSON objects
    sample_json_objects = [
        {
            "context": "Today is a sunny day.",
            "target": "I'm going for a walk in the park.",
        },
        {
            "context": "I just finished a great book.",
            "target": "The ending was unexpected, and I loved it!",
        },
        {
            "context": "It's time for dinner.",
            "target": "I'm going to cook spaghetti tonight.",
        },
    ]

    # File path to save the JSON objects
    file_path = "sample_data.json"

    # Write the JSON objects to the file
    with open(file_path, "w") as file:
        for json_obj in sample_json_objects:
            json_str = json.dumps(json_obj)
            file.write(json_str + "\n")

    data_source = LLMDataSource(file_path)
    dataset_loader = dataset.DatasetLoader(
        data_source=data_source, featurizer=featurizer, shuffle=True
    )
    data = dataset_loader.load_all(1)
    training_inputs, training_labels = (
        bolt.train.convert_datasets(
            data[:-1], dims=[VOCAB_SIZE, VOCAB_SIZE, (2**32) - 1, VOCAB_SIZE]
        ),
        bolt.train.convert_dataset(data[-1], dim=VOCAB_SIZE),
    )
    assert len(training_inputs) == 20 and len(training_labels) == 20


@pytest.mark.unit
def test_unified_llm_data_source():
    sample_json_objects_1 = [
        {
            "context": "Today is a sunny day.",
            "target": "I'm going for a walk in the park.",
        },
        {
            "context": "I just finished a great book.",
            "target": "The ending was unexpected, and I loved it!",
        },
        {
            "context": "It's time for dinner.",
            "target": "I'm going to cook spaghetti tonight.",
        },
    ]

    file_path_1 = "sample_data.json"
    file1_data = []
    # Write the JSON objects to the file
    with open(file_path_1, "w") as file:
        for json_obj in sample_json_objects_1:
            json_str = json.dumps(json_obj)
            file1_data.append(json_str)
            file.write(json_str + "\n")

    sample_json_objects_2 = [
        {
            "context": "The project deadline is approaching.",
            "target": "I need to work extra hours to meet the deadline.",
        },
        {
            "context": "I received a call from an old friend.",
            "target": "We are planning to meet for coffee this weekend.",
        },
        {
            "context": "I enjoy playing musical instruments.",
            "target": "I'm thinking of learning to play the piano next.",
        },
        {
            "context": "The weather forecast predicts rain tomorrow.",
            "target": "I should remember to take an umbrella when I go out.",
        },
        {
            "context": "I attended a virtual conference yesterday.",
            "target": "The keynote speaker gave an inspiring talk.",
        },
        {
            "context": "I have a busy schedule this week.",
            "target": "I need to prioritize my tasks to stay organized.",
        },
        {
            "context": "I'm learning a new programming language.",
            "target": "It's challenging but exciting to explore new technologies.",
        },
        {
            "context": "I enjoy outdoor activities.",
            "target": "Hiking and camping are my favorite weekend activities.",
        },
    ]

    file_path_2 = "sample_data_2.json"
    file2_data = []
    # Write the JSON objects to the file
    with open(file_path_2, "w") as file:
        for json_obj in sample_json_objects_2:
            json_str = json.dumps(json_obj)
            file2_data.append(json_str)
            file.write(json_str + "\n")

    data_source = UnifiedLLMDataSource(
        file_paths=[file_path_1, file_path_2],
        probs=[0.3, 0.7],
        restart_allowed=[True, False],
    )
    line_iterator = data_source._get_line_iterator()

    retrieved_data = []
    for line in line_iterator:
        retrieved_data.append(line.strip())

    num_common_1 = len(set(file1_data).intersection(retrieved_data))
    num_common_2 = len(set(file2_data).intersection(retrieved_data))

    assert num_common_1 / (num_common_1 + num_common_2) >= 0.2
    assert num_common_1 / (num_common_1 + num_common_2) <= 0.4
