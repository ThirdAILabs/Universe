import time

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
            "bert_tokens": np.array(tokens, dtype=np.uint32),
            "doc_offsets": np.array(offsets, dtype=np.uint32),
            "metadata": np.ones(shape=(len(offsets) - 1, METADATA_DIM)),
        }
        input_batches.append(batch)
        label_batches.append(labels[idx : idx + batch_size])

    return input_batches, label_batches, tokenizer.get_vocab_size()


@pytest.fixture
def tokenized_data(download_clinc_dataset):
    train_filename, test_filename, _ = download_clinc_dataset

    train_x, train_y, vocab_size = process_data(train_filename, batch_size=2048)
    test_x, test_y, _ = process_data(test_filename, batch_size=2048)

    return train_x, train_y, test_x, test_y, vocab_size


def test_csdisco_text_classifier_training(tokenized_data):
    train_x, train_y, test_x, test_y, vocab_size = tokenized_data

    model = bolt.UniversalDeepTransformer(
        input_vocab_size=vocab_size,
        metadata_dim=METADATA_DIM,
        n_classes=150,
        model_size="small",
    )

    errors = []
    for _ in range(5):
        s = time.perf_counter()
        for (x, y) in zip(train_x, train_y):
            errors.append(model.train(data=x, labels=y, learning_rate=0.05))
        e = time.perf_counter()
        print("TIME: ", e - s)

    correct = 0
    total = 0
    for (x, y) in zip(test_x, test_y):
        pred = model.predict(x)
        correct += np.sum(np.argmax(pred, axis=1) == np.argmax(y, axis=1))
        total += len(pred)

    print(correct / total)

    # Accuracy is around 0.86-0.88, the gap between this and our regular clinc model 
    # is due to using sigmoid and BCE instead of softmax and CCE.
    assert correct / total >= 0.8
