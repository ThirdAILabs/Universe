import os
import re

import pytest
from test_udt_simple import make_simple_trained_model
from thirdai import bolt, dataset

from conftest import download_bert_base_uncased

pytestmark = [pytest.mark.unit]


TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"


def create_data_file(positive_sample, negative_sample, filename):
    with open(filename, "w") as f:
        f.write("text,category\n")
        f.write(f"{positive_sample},1\n")
        f.write(f"{positive_sample},1\n")
        f.write(f"{negative_sample},0\n")
        f.write(f"{negative_sample},0\n")


def eval_accuracy_and_cleanup(model):
    model.train(TRAIN_FILENAME, epochs=10, learning_rate=0.001)

    metrics = model.evaluate(TEST_FILENAME, metrics=["categorical_accuracy"])

    assert metrics["val_categorical_accuracy"][0] == 1

    save_loc = "temp_save.bolt"
    model.save(save_loc)
    model = bolt.UniversalDeepTransformer.load(save_loc)

    metrics = model.evaluate(TEST_FILENAME, metrics=["categorical_accuracy"])

    assert metrics["val_categorical_accuracy"][0] == 1

    os.remove(TRAIN_FILENAME)
    os.remove(TEST_FILENAME)
    os.remove(save_loc)


@pytest.mark.parametrize("encoding", ["none", "local", "global"])
def test_udt_accepts_valid_text_encodings(encoding):
    make_simple_trained_model(text_encoding_type=encoding)


def test_char_k_text_tokenizer():
    # We want to check if UDT is actually using the character 3 gram block.
    # We do this by memorizing 3 character words then using those words as part
    # of unseen 4 character words in the test data.

    create_data_file("lol", "aya", TRAIN_FILENAME)
    create_data_file("lol9", "aya9", TEST_FILENAME)

    model = bolt.UniversalDeepTransformer(
        data_types={
            "text": bolt.types.text(tokenizer="char-3"),
            "category": bolt.types.categorical(),
        },
        target="category",
        n_target_classes=2,
    )

    eval_accuracy_and_cleanup(model)


def test_words_punct_text_tokenizer():
    # We want to check if UDT is actually using the words-punct tokenizer
    # We do this by passing in words joined with punctuation in the training
    # data then separating them in the testing data

    create_data_file("lol.", "aya?", TRAIN_FILENAME)
    create_data_file("lol .", "aya ?", TEST_FILENAME)

    model = bolt.UniversalDeepTransformer(
        data_types={
            "text": bolt.types.text(tokenizer="words-punct"),
            "category": bolt.types.categorical(),
        },
        target="category",
        n_target_classes=2,
    )

    eval_accuracy_and_cleanup(model)


def test_lowercasing_for_udt_text_type():
    # We want to check if UDT is actually using lowercasing words in the text
    # type. We do this by passing in words with some uppercase characters in the
    # training data then changing the case slightly in the testing data

    create_data_file("Lol", "aYa", TRAIN_FILENAME)
    create_data_file("loL", "ayA", TEST_FILENAME)

    model = bolt.UniversalDeepTransformer(
        data_types={
            "text": bolt.types.text(lowercase=True),
            "category": bolt.types.categorical(),
        },
        target="category",
        n_target_classes=2,
    )

    eval_accuracy_and_cleanup(model)


def test_tokenizer_from_vocabulary(download_bert_base_uncased):
    create_data_file("threading", "foresting", TRAIN_FILENAME)
    create_data_file("thread ##ing", "forest ##ing", TEST_FILENAME)

    BERT_VOCAB_PATH = download_bert_base_uncased
    tokenizer = dataset.WordpieceTokenizer(BERT_VOCAB_PATH)

    model = bolt.UniversalDeepTransformer(
        data_types={
            "text": bolt.types.text(tokenizer=tokenizer),
            "category": bolt.types.categorical(),
        },
        target="category",
        n_target_classes=2,
    )

    eval_accuracy_and_cleanup(model)


def test_hybrid_tokenizer_udt(download_bert_base_uncased):
    create_data_file("threading", "foresting", TRAIN_FILENAME)
    create_data_file("thread ##ing", "forest ##ing", TEST_FILENAME)

    BERT_VOCAB_PATH = download_bert_base_uncased
    tokenizer = dataset.HybridTokenizer(BERT_VOCAB_PATH)

    model = bolt.UniversalDeepTransformer(
        data_types={
            "text": bolt.types.text(tokenizer=tokenizer),
            "category": bolt.types.categorical(),
        },
        target="category",
        n_target_classes=2,
    )

    eval_accuracy_and_cleanup(model)


def test_invalid_text_tokenizers():
    invalid_tokenizer = "INVALID"
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Created text column with invalid tokenizer 'INVALID', please choose one of 'words', 'words-punct', or 'char-k' (k is a number, e.g. 'char-5')."
        ),
    ):
        bolt.UniversalDeepTransformer(
            data_types={
                "text_col": bolt.types.text(tokenizer=invalid_tokenizer),
                "some_random_name": bolt.types.categorical(),
            },
            target="target",
            n_target_classes=2,
        )


def test_contextual_text_encodings():
    invalid_encoding = "INVALID"
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Created text column with invalid contextual_encoding '{invalid_encoding}', please choose one of 'none', 'local', 'ngram-N', or 'global'."
        ),
    ):
        bolt.UniversalDeepTransformer(
            data_types={
                "text_col": bolt.types.text(contextual_encoding=invalid_encoding),
                "some_random_name": bolt.types.categorical(),
            },
            target="target",
            n_target_classes=2,
        )


@pytest.mark.parametrize(
    "invalid_tokenizer", ["char-3p", "char-0", "char-10-0", "char0"]
)
def test_invalid_char_k_tokenizer(invalid_tokenizer):
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Created text column with invalid tokenizer '{invalid_tokenizer}', please choose one of 'words', 'words-punct', or 'char-k' (k is a number, e.g. 'char-5')."
        ),
    ):
        bolt.UniversalDeepTransformer(
            data_types={
                "text_col": bolt.types.text(tokenizer=invalid_tokenizer),
                "some_random_name": bolt.types.categorical(),
            },
            target="target",
            n_target_classes=2,
        )


@pytest.mark.parametrize("invalid_ngram", ["ngram-3p", "ngram0", "ngram-10-0"])
def test_invalid_ngram_encoder(invalid_ngram):
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Created text column with invalid contextual_encoding '{invalid_ngram}', please choose one of 'none', 'local', 'ngram-N', or 'global'."
        ),
    ):
        bolt.UniversalDeepTransformer(
            data_types={
                "text_col": bolt.types.text(contextual_encoding=invalid_ngram),
                "some_random_name": bolt.types.categorical(),
            },
            target="target",
            n_target_classes=2,
        )


def test_invalid_ngram_encoder_n_equals_0():
    invalid_ngram = "ngram-0"
    with pytest.raises(
        ValueError,
        match=re.escape(f"Specified 'ngram-N' option with N = 0. Please use N > 0."),
    ):
        bolt.UniversalDeepTransformer(
            data_types={
                "text_col": bolt.types.text(contextual_encoding=invalid_ngram),
                "some_random_name": bolt.types.categorical(),
            },
            target="target",
            n_target_classes=2,
        )
