import json
import os
import random
import re
import string

import pandas as pd
import pytest
from faker import Faker
from thirdai import bolt, data, dataset

pytestmark = [pytest.mark.unit, pytest.mark.release]

TOKENS = "tokens"
TAGS = "tags"

fake = Faker(seed=0)


ner_whitespace_formatting_sample = """Subject: Catching up!

Hey John Brown,

It's been a while since we've chatted! How are things going with you?

I'm finally starting to feel settled in my new place at 646 Dunberry Road. It took a while to unpack everything (I swear I have more boxes than Albert Schweitzer has in their entire house!). I've been thinking about taking a drive over to your place sometime soon, just to catch up properly. I could swing by on my birthday: 05/04/2024, after my afternoon shift at work. What do you think? Maybe we could grab a bite to eat at Los Pollos Hermanos or something.

Oh, and have you had a chance to check out my new website, www.sophiaseamstress.com? It's still under construction but I'm slowly getting the hang of it. It'd be great to hear what you think!

By the way, remember that Peruvian restaurant that you suggested I try near your place at 78 Seville Street? I went there the other day with Jane Docent and it was absolutely delicious! I can't wait to tell you all about it when I see you!

Speaking of things that I need to tell you about, I actually had this really weird experience the other day. I was at the DMV trying to renew my driver's license, and you wouldn't believe the mix-up! They were trying to tell me my license number was A8734433, but I knew it was actually D8453436. It was so frustrating, it took ages to get sorted out. Can you believe that?!

I should probably head out for a quick run now. My gym membership, at Space Jam Gym, hasn't really paid off yet, but I'm determined to reach my fitness goals. Hopefully I can beat your record this time, Mr John Brown the marathon winner!

Let me know when you're free.

Talk soon!

Sophia Alexander

408 374 5722
sophia@goodmail.com
"""


def random_credit_card():
    number_str = fake.credit_card_number(card_type="visa")
    return number_str


def random_email():
    return (
        "".join(random.choices(string.ascii_letters, k=5))
        + "@"
        + random.choice(["gmail.com", "hotmail.net", "outlook.com"])
    )


def generate_sample():
    email_tokens = ["email", "is", random_email(), "for", "work"]
    email_tags = ["O", "O", "EMAIL", "O", "O"]

    credit_card_tokens = ["credit", "card", "is", random_credit_card()]
    credit_card_tags = ["O", "O", "O", "CREDITCARDNUMBER"]

    return (
        " ".join(email_tokens + credit_card_tokens),
        " ".join(email_tags + credit_card_tags),
    )


def generate_data(filename, n_rows):
    with open(filename, "w") as file:
        file.write(f"{TOKENS},{TAGS}\n")
        for _ in range(n_rows):
            source_str, target_str = generate_sample()
            sample = source_str + "," + target_str

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


def evaluate_with_offsets(model, test):
    correct = 0
    total = 0

    samples, labels = load_eval_samples(test)

    all_predictions_with_offsets = model.predict_batch(samples, return_offsets=True)

    for sample, predicted_entities, expected_tags in zip(
        samples, all_predictions_with_offsets, labels
    ):
        tokens = sample[TOKENS].split()
        predicted_tags = ["O"] * len(tokens)

        for entity in predicted_entities:
            start = entity["BeginOffset"]
            end = entity["EndOffset"]
            predicted_tag = entity["Type"]

            for i, token in enumerate(tokens):
                token_start = sum(len(t) + 1 for t in tokens[:i])
                token_end = token_start + len(token)
                if token_start >= start and token_end <= end:
                    predicted_tags[i] = predicted_tag

        assert len(predicted_tags) == len(expected_tags)
        for predicted_tag, expected_tag in zip(predicted_tags, expected_tags):
            if expected_tag != "O":
                if predicted_tag == expected_tag:
                    correct += 1
                total += 1

    return correct / total if total > 0 else 0.0


def evaluate(model, test):
    predict_acc = evaluate_predict(model, test)
    predict_batch_acc = evaluate_predict_batch(model, test)
    predict_offset_acc = evaluate_with_offsets(model, test)

    assert predict_acc == predict_batch_acc

    assert predict_acc == predict_offset_acc

    return predict_acc


@pytest.mark.parametrize(
    "use_rules,ignore_rule_tags",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_udt_ner_model(ner_dataset, use_rules, ignore_rule_tags):
    train, test = ner_dataset

    model = bolt.UniversalDeepTransformer(
        data_types={
            TOKENS: bolt.types.text(),
            TAGS: bolt.types.token_tags(
                tags=["EMAIL", "CREDITCARDNUMBER"], default_tag="O"
            ),
        },
        target=TAGS,
        embedding_dimension=500,
        rules=use_rules,
        ignore_rule_tags=ignore_rule_tags,
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

    # check splitting on whitespaces
    predicted_tags = model.predict({"tokens": ner_whitespace_formatting_sample})
    assert len(predicted_tags) == len(ner_whitespace_formatting_sample.split())


def test_udt_ner_from_pretrained(ner_dataset):
    train, test = ner_dataset

    pretrained_model = bolt.UniversalDeepTransformer(
        data_types={
            TOKENS: bolt.types.text(),
            TAGS: bolt.types.token_tags(
                tags=["EMAIL", "CREDITCARDNUMBER"], default_tag="O"
            ),
        },
        target=TAGS,
        embedding_dimension=450,
    )

    model = bolt.UniversalDeepTransformer(
        data_types={
            TOKENS: bolt.types.text(),
            TAGS: bolt.types.token_tags(
                tags=["EMAIL", "CREDITCARDNUMBER"], default_tag="O"
            ),
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


def test_udt_ner_learned_tags(ner_dataset):
    train, _ = ner_dataset

    rule_tags = ["CREDITCARDNUMBER"]

    # supported_type is set to int to verify that the filters are being used in the prediction pipeline
    learned_tags = [
        data.transformations.NERLearnedTag(
            "EMAIL", supported_type="int", consecutive_tags_required=2
        )
    ]

    model = bolt.UniversalDeepTransformer(
        data_types={
            TOKENS: bolt.types.text(),
            TAGS: bolt.types.token_tags(tags=rule_tags + learned_tags, default_tag="O"),
        },
        target=TAGS,
        embedding_dimension=500,
        rules=True,
        use_token_tag_counter=True,
    )

    model.train(train, epochs=1, learning_rate=0.001, metrics=["categorical_accuracy"])

    samples = [{"tokens": generate_sample()[0]} for _ in range(100)]
    predictions = model.predict_batch(samples, top_k=1)

    for prediction in predictions:
        for tags in prediction:
            assert tags[0][0] != "EMAIL"

    model.save("udt_ner_model.bolt")
    model = bolt.UniversalDeepTransformer.load("udt_ner_model.bolt")

    new_predictions = model.predict_batch(samples, top_k=1)

    for old_prediction, new_prediction in zip(predictions, new_predictions):
        for old_tag_prediction, new_tag_prediction in zip(
            old_prediction, new_prediction
        ):
            assert old_tag_prediction[0][0] == new_tag_prediction[0][0]


def test_udt_ner_add_new_tag(ner_dataset):
    train, _ = ner_dataset

    model = bolt.UniversalDeepTransformer(
        data_types={
            TOKENS: bolt.types.text(),
            TAGS: bolt.types.token_tags(
                tags=["EMAIL", "CREDITCARDNUMBER"], default_tag="O"
            ),
        },
        target=TAGS,
    )
    model.train(train, epochs=1, learning_rate=0.001)

    # add a new entity to the model
    model.add_ner_entities(["NAME"])

    # generate temp dataset
    data = pd.DataFrame(
        {TOKENS: ["My name is ABC"] * 1000, TAGS: ["O O O NAME"] * 1000}
    )
    data.to_csv("temp_name_tag_file.csv", index=False)
    model.train("temp_name_tag_file.csv", epochs=2, learning_rate=0.001)

    predictions = model.predict({TOKENS: "My name is ABC"}, top_k=1)
    assert [pred[0][0] for pred in predictions] == ["O", "O", "O", "NAME"]

    os.remove("temp_name_tag_file.csv")
