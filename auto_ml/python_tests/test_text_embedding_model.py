import numpy as np
import pandas as pd
import pytest
from thirdai import bolt, dataset

pytextmark = [pytest.mark.unit]


def get_distance(embedding_model, string_1, string_2):
    embedding_1 = embedding_model.encode(string_1).activations.flatten()
    embedding_2 = embedding_model.encode(string_2).activations.flatten()
    return np.linalg.norm(embedding_1 - embedding_2)


def test_basic_text_embedding_model():
    model = bolt.UniversalDeepTransformer(
        data_types={
            "strong": bolt.types.text(contextual_encoding="char-3"),
            "index": bolt.types.categorical(),
        },
        target="index",
        n_target_classes=100,
        integer_target=True,
        options={"extreme_classification": True, "embedding_dimension": 256},
    )

    embedding_model = model.get_text_embedding_model()
    string_1 = "this is"
    string_2 = "sparta"
    distance_1 = get_distance(embedding_model, string_1, string_2)

    num_examples = 1000
    training_data = {
        "input_1": [string_1] * num_examples,
        "input_2": [string_2] * num_examples,
        "labels": [1] * num_examples,
    }
    pd.DataFrame(training_data).to_csv("easy_contrastive_sparta_data.csv", index=False)

    embedding_model.supervised_train(
        dataset.FileDataSource("easy_contrastive_sparta_data.csv"),
        input_col_1="input_1",
        input_col_2="input_2",
        label_col="labels",
        learning_rate=0.001,
        epochs=30,
    )
    distance_2 = get_distance(embedding_model, string_1, string_2)

    embedding_model.save("embedding_model.bin")

    embedding_model = bolt.TextEmbeddingModel.load("embedding_model.bin")
    distance_3 = get_distance(embedding_model, string_1, string_2)

    print(distance_1, distance_2, distance_3)
    assert distance_1 > distance_2
    assert distance_2 == distance_3
