import pytest
from thirdai.dataset import PretrainingTextDataSource
from thirdai import bolt, dataset


@pytest.mark.unit
def test_llm_pretraining_data_source():
    VOCAB_SIZE = 50267
    DUMMY_FILE = "dummy_text.txt"

    featurizer = dataset.TextGenerationFeaturizer(
        lrc_len=3, irc_len=2, src_len=1, vocab_size=VOCAB_SIZE
    )

    dummy_text = """Hello world!
    This is a test.
    Generative models are fascinating."""

    with open(DUMMY_FILE, "w") as f:
        f.write(dummy_text)

    data_source = PretrainingTextDataSource(DUMMY_FILE)
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
    assert len(training_inputs) == 9 and len(training_labels) == 9
