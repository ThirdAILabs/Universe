import json
import os
import random
import re
import string

import pytest
import pandas as pd

from thirdai import bolt, data, dataset

pytestmark = [pytest.mark.unit, pytest.mark.release]

TOKENS = "tokens"
TAGS = "tags"


def random_credit_card():
    return "-".join("".join(random.choices("0123456789", k=4)) for _ in range(4))


def random_email():
    return (
        "".join(random.choices(string.ascii_letters, k=5))
        + "@"
        + random.choice(["gmail.com", "hotmail.net", "outlook.com"])
    )


def generate_data(filename, n_rows):
    with open(filename, "w") as file:
        file.write(f"{TOKENS},{TAGS}\n")
        for _ in range(n_rows):
            email_tokens = ["email", "is", random_email(), "for", "work"]
            email_tags = ["O", "O", "email", "O", "O"]

            credit_card_tokens = ["credit", "card", "is", random_credit_card()]
            credit_card_tags = ["O", "O", "O", "credit_card"]

            sample = (
                " ".join(email_tokens + credit_card_tokens)
                + ","
                + " ".join(email_tags + credit_card_tags)
            )

            file.write(sample + "\n")


@pytest.fixture(scope="session")
def ner_dataset():
    train_file = "simple_ner_train.jsonl"
    test_file = "simple_ner_test.jsonl"

    generate_data(train_file, 10000)
    generate_data(test_file, 100)

    yield train_file, test_file

    for file in [train_file, test_file]:
        if os.path.exists(file):
            os.remove(file)


def load_eval_samples(test):
    samples = []
    labels = []
    df = pd.read_csv(test)
    for tokens, tags in zip(df[TOKENS], df[TAGS]):
        samples.append({TOKENS: tokens})
        labels.append(tags.split(" "))
    return samples, labels


def evaluate_predict(model, test):
    correct = 0
    total = 0

    samples, labels = load_eval_samples(test)
    for sample, expected_tags in zip(samples, labels):
        predicted_tags = model.predict(sample)
        predicted_tags = [x[0][0] for x in predicted_tags]

        assert len(predicted_tags) == len(expected_tags)
        for tag, expected_tag in zip(predicted_tags, expected_tags):
            if expected_tag != "O":
                if tag == expected_tag:
                    correct += 1
                total += 1

    return correct / total


def evaluate_predict_batch(model, test):
    correct = 0
    total = 0

    samples, labels = load_eval_samples(test)

    all_predicted_tags = model.predict_batch(samples)

    for predicted_tags, expected_tags in zip(all_predicted_tags, labels):
        predicted_tags = [x[0][0] for x in predicted_tags]

        assert len(predicted_tags) == len(expected_tags)
        for tag, expected_tag in zip(predicted_tags, expected_tags):
            if expected_tag != "O":
                if tag == expected_tag:
                    correct += 1
                total += 1

    return correct / total


def evaluate(model, test):
    predict_acc = evaluate_predict(model, test)
    predict_batch_acc = evaluate_predict_batch(model, test)

    assert predict_acc == predict_batch_acc

    return predict_acc


def test_udt_ner(ner_dataset):
    train, test = ner_dataset

    model = bolt.UniversalDeepTransformer(
        data_types={
            TOKENS: bolt.types.text(),
            TAGS: bolt.types.token_tags(tags=["email", "credit_card"], default_tag="O"),
        },
        target=TAGS,
        embedding_dimension=500,
    )

    model.train(train, epochs=1, learning_rate=0.001, metrics=["categorical_accuracy"])

    metrics = model.evaluate(test, metrics=["categorical_accuracy"])

    assert metrics["val_categorical_accuracy"][-1] >= 0.9

    acc_before_save = evaluate(model, test)
    print(f"{acc_before_save=}")
    assert acc_before_save > 0.9

    save_path = "udt_ner_model.bolt"
    model.save(save_path)

    model = bolt.UniversalDeepTransformer.load(save_path)
    os.remove(save_path)

    acc_after_load = evaluate(model, test)
    print(f"{acc_after_load=}")
    assert acc_after_load > 0.9

    model.train(test, epochs=1, learning_rate=0.001)

    acc_after_finetune = evaluate(model, test)
    print(f"{acc_after_finetune=}")
    assert acc_after_finetune > 0.9


def test_udt_ner_from_pretrained(ner_dataset):
    train, test = ner_dataset

    pretrained_model = bolt.UniversalDeepTransformer(
        data_types={
            TOKENS: bolt.types.text(),
            TAGS: bolt.types.token_tags(tags=["email", "credit_card"], default_tag="O"),
        },
        target=TAGS,
        embedding_dimension=450,
    )

    model = bolt.UniversalDeepTransformer(
        data_types={
            TOKENS: bolt.types.text(),
            TAGS: bolt.types.token_tags(tags=["email", "credit_card"], default_tag="O"),
        },
        target=TAGS,
        pretrained_model=pretrained_model,
    )

    assert model.model_dims()[1] == 450

    model.train(train, epochs=1, learning_rate=0.001, metrics=["categorical_accuracy"])

    metrics = model.evaluate(test, metrics=["categorical_accuracy"])

    assert metrics["val_categorical_accuracy"][-1] >= 0.9


def test_udt_ner_target_tokenizer_arg():
    bolt.UniversalDeepTransformer(
        data_types={
            TOKENS: bolt.types.text(),
            TAGS: bolt.types.token_tags(tags=["email", "credit_card"], default_tag="O"),
        },
        target=TAGS,
        target_tokenizers=[dataset.CharKGramTokenizer(3)],
    )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Invalid type for argument 'target_tokenizers'. Must be either List[int] or List[dataset.Tokenizer]."
        ),
    ):
        bolt.UniversalDeepTransformer(
            data_types={
                TOKENS: bolt.types.text(),
                TAGS: bolt.types.token_tags(
                    tags=["email", "credit_card"], default_tag="O"
                ),
            },
            target=TAGS,
            target_tokenizers=[dataset.NGramEncoder(2)],
        )


def test_udt_ner_feature_config_arg():
    bolt.UniversalDeepTransformer(
        data_types={
            TOKENS: bolt.types.text(),
            TAGS: bolt.types.token_tags(tags=["email", "credit_card"], default_tag="O"),
        },
        target=TAGS,
        feature_config=data.transformations.NerFeatureConfig(*([True] * 7)),
    )

    with pytest.raises(ValueError, match="Invalid type .*"):
        bolt.UniversalDeepTransformer(
            data_types={
                TOKENS: bolt.types.text(),
                TAGS: bolt.types.token_tags(
                    tags=["email", "credit_card"], default_tag="O"
                ),
            },
            target=TAGS,
            feature_config={"email": True},
        )
