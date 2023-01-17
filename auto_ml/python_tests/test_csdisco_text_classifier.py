import numpy as np
import pandas as pd
import pytest
from download_dataset_fixtures import download_clinc_dataset
from thirdai import bolt
from tokenizers import Tokenizer

METADATA_DIM = 10


def process_data(filename, batch_size):
    df = pd.read_csv(filename).sample(frac=1.0, random_state=8)

    n_classes = len(df["category"].unique())

    original_labels = df["category"].to_numpy()
    labels = np.eye(n_classes, dtype=np.float32)[original_labels]

    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")

    encodings = tokenizer.encode_batch(df["text"].iloc)

    input_batches = []
    label_batches = []
    for idx in range(0, len(encodings), batch_size):
        tokens = []
        offsets = [0]
        for e in encodings[idx : idx + batch_size]:
            tokens.extend(e.ids[1:-1])
            offsets.append(len(tokens))

        batch = {
            "tokens": np.array(tokens, dtype=np.uint32),
            "offsets": np.array(offsets, dtype=np.uint32),
            "metadata": np.ones(shape=(len(offsets) - 1, METADATA_DIM)),
        }
        input_batches.append(batch)
        label_batches.append(labels[idx : idx + batch_size])

    return input_batches, label_batches, tokenizer.get_vocab_size()


@pytest.fixture(scope="session")
def tokenized_data(download_clinc_dataset):
    train_filename, test_filename, _ = download_clinc_dataset

    train_x, train_y, vocab_size = process_data(train_filename, batch_size=2048)
    test_x, test_y, _ = process_data(test_filename, batch_size=2048)

    return train_x, train_y, test_x, test_y, vocab_size


def compute_bce_loss(scores, labels):
    scores = np.clip(scores, 1e-6, 1 - 1e-6)
    losses = labels * np.log(scores) + (1 - labels) * np.log(1 - scores)

    return -np.mean(losses), -np.mean(losses, axis=0)


def train_epoch(model, train_x, train_y, learning_rate=0.05):
    for (x, y) in zip(train_x, train_y):
        val_loss = model.validate(data=x, labels=y)

        scores = model.predict(data=x)

        avg_loss, class_loss = compute_bce_loss(scores, y)

        avg_loss_train = model.train(data=x, labels=y, learning_rate=learning_rate)

        assert np.allclose([avg_loss_train], avg_loss, atol=1e-5)
        assert np.allclose([val_loss["mean_loss"]], avg_loss, atol=1e-5)
        assert np.allclose(val_loss["per_class_loss"], class_loss, atol=1e-5)


def accuracy(model, test_x, test_y):
    correct = 0
    total = 0
    for (x, y) in zip(test_x, test_y):
        pred = model.predict(x)
        correct += np.sum(np.argmax(pred, axis=1) == np.argmax(y, axis=1))
        total += len(pred)

    return correct / total


@pytest.fixture(scope="session")
def train_model(tokenized_data):
    train_x, train_y, _, _, vocab_size = tokenized_data

    model = bolt.UniversalDeepTransformer(
        input_vocab_size=vocab_size,
        metadata_dim=METADATA_DIM,
        n_classes=150,
        model_size="small",
    )

    for _ in range(5):
        train_epoch(model, train_x, train_y)

    return model


def test_text_classifier_training(train_model, tokenized_data):
    _, _, test_x, test_y, _ = tokenized_data

    model = train_model

    # Accuracy is around 0.86-0.88, the gap between this and our regular clinc model
    # is due to using sigmoid and BCE instead of softmax and CCE.
    assert accuracy(model, test_x, test_y) >= 0.8


def test_text_classifier_load_save(train_model, tokenized_data):
    train_x, train_y, test_x, test_y, _ = tokenized_data
    model = train_model

    path = "./saved_text_classifier.bolt"
    model.save(path)

    model = bolt.UniversalDeepTransformer.load(path)

    assert accuracy(model, test_x, test_y) >= 0.8

    train_epoch(model, train_x, train_y, learning_rate=0.01)

    assert accuracy(model, test_x, test_y) >= 0.8
