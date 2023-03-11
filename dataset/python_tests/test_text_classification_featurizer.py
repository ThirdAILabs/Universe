import pytest
from thirdai.dataset import (
    DatasetLoader,
    FileDataSource,
    TextGenerationFeaturizer,
    TextClassificationFeaturizer,
    TextTokens,
)
import numpy as np


OTHER_FILENAME = "other_dummy.csv"
FILENAME = "dummy.csv"
OUTPUT_CLASSES = 3
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
DELIM = ","
TOKENS = [
    np.array([8]),
    np.array([8, 200]),
    np.array([8, 200, 332]),
    np.array([8, 200, 332, 1]),
    np.array([8, 200, 332, 1, 55]),
]
LABEL_DELIM = " "
LABELS = [
    np.array(["x", "y"]),
    np.array(["z"]),
    np.array(["x"]),
    np.array(["y"]),
    np.array(["z"]),
]


@pytest.fixture
def generate_dummy_data():
    with open(FILENAME, "w") as out:
        out.write(f"{TEXT_COLUMN}{DELIM}{LABEL_COLUMN}\n")
        for tokens, labels in zip(TOKENS, LABELS):
            out.write(
                f"{' '.join(map(str, tokens))}{DELIM}{LABEL_DELIM.join(labels)}\n"
            )

    with open(OTHER_FILENAME, "w") as out:
        out.write(f"{' '.join(map(str, TOKENS[-1])) + ' 0'}\n")


@pytest.mark.unit
def test_text_classification_featurizer(generate_dummy_data):
    featurizer = TextClassificationFeaturizer(
        text_column=TEXT_COLUMN,
        label_column=LABEL_COLUMN,
        delimiter=DELIM,
        n_labels=3,
        tokens=TextTokens.UNI_PAIR,
        label_delimiter=LABEL_DELIM,
        integer_labels=False,
        normalize_categories=False,
    )
    pipeline = DatasetLoader(
        data_source=FileDataSource(FILENAME), featurizer=featurizer, shuffle=False
    )
    [unigrams, pairgrams, labels] = pipeline.load_all(batch_size=256)

    for i in range(5):
        labels_i = np.array(
            [featurizer.label_from_id(uid) for uid in labels[0][i].to_numpy()[0]]
        )
        assert (labels_i == LABELS[i]).all()

    og_featurizer = TextGenerationFeaturizer(
        lrc_len=5,
        irc_len=5,
        src_len=5,
    )
    og_pipeline = DatasetLoader(
        data_source=FileDataSource(OTHER_FILENAME),
        featurizer=og_featurizer,
        shuffle=False,
    )
    [og_unigrams, og_pairgrams, _, _] = og_pipeline.load_all(batch_size=256)

    for i in range(5):
        assert unigrams[0][i].__str__() == og_unigrams[0][i].__str__()
    for i in range(5):
        assert pairgrams[0][i].__str__() == og_pairgrams[0][i].__str__()
