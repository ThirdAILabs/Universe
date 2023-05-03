import os

import pytest
from test_udt_simple import make_simple_trained_model
from thirdai import bolt, dataset

from conftest import download_bert_tokenizer

pytestmark = [pytest.mark.unit]


def eval_accuracy_and_cleanup(model, train_filename, test_filename):
    model.train(train_filename, epochs=10, learning_rate=0.001)

    metrics = model.evaluate(
        test_filename, return_metrics=True, metrics=["categorical_accuracy"]
    )

    assert metrics["categorical_accuracy"] == 1

    save_loc = "temp_save.bolt"
    model.save(save_loc)
    model = bolt.UniversalDeepTransformer.load(save_loc)

    metrics = model.evaluate(
        test_filename, return_metrics=True, metrics=["categorical_accuracy"]
    )

    assert metrics["categorical_accuracy"] == 1

    os.remove(train_filename)
    os.remove(test_filename)
    os.remove(save_loc)


@pytest.mark.parametrize("encoding", ["none", "local", "global"])
def test_udt_accepts_valid_text_encodings(encoding):
    make_simple_trained_model(text_encoding_type=encoding)


def test_char_k_text_tokenizer():
    # We want to check if UDT is actually using the character 3 gram block.
    # We do this by memorizing 3 character words then using those words as part
    # of unseen 4 character words in the test data.

    train_filename = "train.csv"
    with open(train_filename, "w") as f:
        f.write("text,category\n")
        f.write("lol,1\n")
        f.write("lol,1\n")
        f.write("aya,0\n")
        f.write("aya,0\n")

    test_filename = "test.csv"
    with open(test_filename, "w") as f:
        f.write("text,category\n")
        f.write("lol9,1\n")
        f.write("lol9,1\n")
        f.write("aya9,0\n")
        f.write("aya9,0\n")

    model = bolt.UniversalDeepTransformer(
        data_types={
            "text": bolt.types.text(tokenizer="char-3"),
            "category": bolt.types.categorical(),
        },
        target="category",
        n_target_classes=2,
    )

    eval_accuracy_and_cleanup(model, train_filename, test_filename)


def test_words_punct_text_tokenizer():
    # We want to check if UDT is actually using the words-punct tokenizer
    # We do this by passing in words joined with punctuation in the training
    # data then separating them in the testing data

    train_filename = "train.csv"
    with open(train_filename, "w") as f:
        f.write("text,category\n")
        f.write("lol.,1\n")
        f.write("lol.,1\n")
        f.write("aya?,0\n")
        f.write("aya?,0\n")

    test_filename = "test.csv"
    with open(test_filename, "w") as f:
        f.write("text,category\n")
        f.write("lol .,1\n")
        f.write("lol .,1\n")
        f.write("aya ?,0\n")
        f.write("aya ?,0\n")

    model = bolt.UniversalDeepTransformer(
        data_types={
            "text": bolt.types.text(tokenizer="words-punct"),
            "category": bolt.types.categorical(),
        },
        target="category",
        n_target_classes=2,
    )

    eval_accuracy_and_cleanup(model, train_filename, test_filename)


def test_lowercasing_for_udt_text_type():
    # We want to check if UDT is actually using lowercasing words in the text
    # type. We do this by passing in words with some uppercase characters in the
    # training data then changing the case slightly in the testing data

    train_filename = "train.csv"
    with open(train_filename, "w") as f:
        f.write("text,category\n")
        f.write("Lol,1\n")
        f.write("lOl,1\n")
        f.write("Aya,0\n")
        f.write("aYa,0\n")

    test_filename = "test.csv"
    with open(test_filename, "w") as f:
        f.write("text,category\n")
        f.write("loL,1\n")
        f.write("loL,1\n")
        f.write("ayA,0\n")
        f.write("ayA,0\n")

    model = bolt.UniversalDeepTransformer(
        data_types={
            "text": bolt.types.text(lowercase=True),
            "category": bolt.types.categorical(),
        },
        target="category",
        n_target_classes=2,
    )

    eval_accuracy_and_cleanup(model, train_filename, test_filename)


def test_tokenizer_from_vocabulary(download_bert_tokenizer):
    train_filename = "train.csv"
    with open(train_filename, "w") as f:
        f.write("text,category\n")
        f.write("threading,1\n")
        f.write("threading,1\n")
        f.write("foresting,0\n")
        f.write("foresting,0\n")

    test_filename = "test.csv"
    with open(test_filename, "w") as f:
        f.write("text,category\n")
        f.write("thread ##ing,1\n")
        f.write("thread ##ing,1\n")
        f.write("forest ##ing,0\n")
        f.write("forest ##ing,0\n")

    BERT_VOCAB_PATH = download_bert_tokenizer
    tokenizer = dataset.Wordpiece(BERT_VOCAB_PATH)

    model = bolt.UniversalDeepTransformer(
        data_types={
            "text": bolt.types.text(tokenizer=tokenizer),
            "category": bolt.types.categorical(),
        },
        target="category",
        n_target_classes=2,
    )

    eval_accuracy_and_cleanup(model, train_filename, test_filename)
