import json

import pytest
from thirdai import bolt, dataset
from thirdai.dataset import LLMDataSource


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
