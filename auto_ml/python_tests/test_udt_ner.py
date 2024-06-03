import json
import os
import random
import re
import string

import pytest
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
        for _ in range(n_rows):
            email_tokens = ["email", "is", random_email(), "for", "work"]
            email_tags = ["0", "0", "email", "0", "0"]

            credit_card_tokens = ["credit", "card", "is", random_credit_card()]
            credit_card_tags = ["0", "0", "0", "credit_card"]

            sample = {
                TOKENS: email_tokens + credit_card_tokens,
                TAGS: email_tags + credit_card_tags,
            }

            file.write(json.dumps(sample) + "\n")


@pytest.fixture(scope="session")
def ner_dataset():
    train_file = "simple_ner_train.jsonl"
    test_file = "simple_ner_test.jsonl"

    generate_data(train_file, 10000)
    generate_data(test_file, 20)

    yield train_file, test_file

    for file in [train_file, test_file]:
        if os.path.exists(file):
            os.remove(file)


def evaluate(model, test):
    correct = 0
    total = 0
    for line in open(test):
        data = json.loads(line)

        predicted_tags = model.predict({TOKENS: " ".join(data[TOKENS])})
        predicted_tags = [x[0][0] for x in predicted_tags]

        assert len(predicted_tags) == len(data[TAGS])
        for tag, expected_tag in zip(predicted_tags, data[TAGS]):
            if expected_tag != "0":
                if tag == expected_tag:
                    correct += 1
                total += 1

    return correct / total


def test_udt_ner(ner_dataset):
    train, test = ner_dataset

    model = bolt.UniversalDeepTransformer(
        data_types={
            TOKENS: bolt.types.text(),
            TAGS: bolt.types.token_tags(tags=["0", "email", "credit_card"]),
        },
        target=TAGS,
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


def test_udt_ner_target_tokenizer_arg():
    bolt.UniversalDeepTransformer(
        data_types={
            TOKENS: bolt.types.text(),
            TAGS: bolt.types.token_tags(tags=["0", "email", "credit_card"]),
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
                TAGS: bolt.types.token_tags(tags=["0", "email", "credit_card"]),
            },
            target=TAGS,
            target_tokenizers=[dataset.NGramEncoder(2)],
        )


def test_udt_ner_feature_config_arg():
    bolt.UniversalDeepTransformer(
        data_types={
            TOKENS: bolt.types.text(),
            TAGS: bolt.types.token_tags(tags=["0", "email", "credit_card"]),
        },
        target=TAGS,
        feature_config=data.transformations.NerFeatureConfig(*([True] * 7)),
    )

    with pytest.raises(ValueError, match="Invalid type .*"):
        bolt.UniversalDeepTransformer(
            data_types={
                TOKENS: bolt.types.text(),
                TAGS: bolt.types.token_tags(tags=["0", "email", "credit_card"]),
            },
            target=TAGS,
            feature_config={"email": True},
        )
