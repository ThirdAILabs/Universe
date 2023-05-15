import os
from collections import defaultdict

import pandas as pd
import pytest
from download_dataset_fixtures import download_scifact_dataset
from thirdai import bolt, dataset

pytestmark = [pytest.mark.unit]


SIMPLE_TEST_FILE = "mach_udt_test.csv"
OUTPUT_DIM = 100


def make_simple_test_file(invalid_data=False):
    with open(SIMPLE_TEST_FILE, "w") as f:
        f.write("text,label\n")
        f.write("haha one time,0\n")
        f.write("haha two times,1\n")
        f.write("haha thrice occurances,2\n")
        if invalid_data:
            f.write("haha,3\n")


def train_simple_mach_udt(integer_target=False, invalid_data=False, embedding_dim=256):
    make_simple_test_file(invalid_data=invalid_data)

    model = bolt.UniversalDeepTransformer(
        data_types={
            "text": bolt.types.text(contextual_encoding="local"),
            "label": bolt.types.categorical(),
        },
        target="label",
        n_target_classes=3,
        integer_target=integer_target,
        options={
            "extreme_classification": True,
            "embedding_dimension": embedding_dim,
            "extreme_output_dim": OUTPUT_DIM,
        },
    )

    model.train(SIMPLE_TEST_FILE, epochs=5, learning_rate=0.001)

    os.remove(SIMPLE_TEST_FILE)

    return model


def calculate_precision(all_relevant_documents, all_recommended_documents, at=1):
    assert len(all_relevant_documents) == len(all_recommended_documents)

    precision = 0
    for relevant_documents, recommended_documents in zip(
        all_relevant_documents, all_recommended_documents
    ):
        score = 0
        for pred_doc_id in recommended_documents[:at]:
            if pred_doc_id in relevant_documents:
                score += 1
        score /= at
        precision += score

    return precision / len(all_recommended_documents)


def get_relevant_documents(supervised_tst_file):
    relevant_documents = []
    with open(supervised_tst_file, "r") as f:
        lines = f.readlines()[1:]
        for line in lines:
            query, doc_ids = line.split(",")
            doc_ids = [int(doc_id.strip()) for doc_id in doc_ids.split(":")]
            relevant_documents.append(doc_ids)

    return relevant_documents


def evaluate_model(model, supervised_tst):
    test_df = pd.read_csv(supervised_tst)
    test_samples = [{"QUERY": text} for text in test_df["QUERY"].tolist()]

    output = model.predict_batch(test_samples)

    all_recommended_documents = []
    for sample in output:
        all_recommended_documents.append([int(doc) for doc, score in sample])

    all_relevant_documents = get_relevant_documents(supervised_tst)

    precision = calculate_precision(all_relevant_documents, all_recommended_documents)

    return precision


def train_on_scifact(download_scifact_dataset, integer_target, coldstart):
    (
        unsupervised_file,
        supervised_trn,
        supervised_tst,
        n_target_classes,
    ) = download_scifact_dataset

    model = bolt.UniversalDeepTransformer(
        data_types={
            "QUERY": bolt.types.text(contextual_encoding="local"),
            "DOC_ID": bolt.types.categorical(delimiter=":"),
        },
        target="DOC_ID",
        n_target_classes=n_target_classes,
        integer_target=integer_target,
        options={"extreme_classification": True, "embedding_dimension": 1024},
    )

    if coldstart:
        metrics = model.cold_start(
            filename=unsupervised_file,
            strong_column_names=["TITLE"],
            weak_column_names=["TEXT"],
            learning_rate=0.001,
            epochs=5,
            metrics=[
                "precision@1",
                "recall@10",
            ],
        )
        assert metrics["train_precision@1"][-1] > 0.90

    validation = bolt.Validation(
        supervised_tst,
        metrics=["precision@1"],
    )

    metrics = model.train(
        filename=supervised_trn,
        learning_rate=0.001 if coldstart else 0.0005,
        epochs=10,
        metrics=[
            "precision@1",
            "recall@10",
        ],
        validation=validation,
    )

    return model, metrics, supervised_tst


def test_mach_udt_on_scifact(download_scifact_dataset):
    model, metrics, supervised_tst = train_on_scifact(
        download_scifact_dataset,
        integer_target=True,
        coldstart=True,
    )

    assert metrics["train_precision@1"][-1] > 0.45

    before_save_precision = evaluate_model(model, supervised_tst)

    assert before_save_precision > 0.45

    save_loc = "model.bolt"
    model.save(save_loc)
    model = bolt.UniversalDeepTransformer.load(save_loc)

    after_save_precision = evaluate_model(model, supervised_tst)

    assert before_save_precision == after_save_precision

    os.remove(save_loc)


# We can't coldstart without integer target but we can still train on the
# supervised data. Asserting an accuracy threshold seems to be very flaky so we
# just assert that we don't run into any failures with string target. The remaining behaviours should be covered by the remaining python and c++ tests
def test_mach_udt_string_target(download_scifact_dataset):
    _, string_metrics, supervised_tst = train_on_scifact(
        download_scifact_dataset, integer_target=False, coldstart=False
    )

    _, integer_metrics, _ = train_on_scifact(
        download_scifact_dataset, integer_target=True, coldstart=False
    )


def test_mach_udt_integer_target_label_too_large():
    with pytest.raises(
        ValueError,
        match=r"Received unexpected label: 3.",
    ):
        train_simple_mach_udt(integer_target=True, invalid_data=True)


@pytest.mark.parametrize(
    "embedding_dim, integer_label",
    [(128, True), (128, False), (256, True), (256, False)],
)
def test_mach_udt_entity_embedding(embedding_dim, integer_label):
    model = train_simple_mach_udt(
        integer_target=integer_label, embedding_dim=embedding_dim
    )
    output_labels = [0, 1] if integer_label else ["0", "1"]
    for output_id, output_label in enumerate(output_labels):
        embedding = model.get_entity_embedding(output_label)
        assert embedding.shape == (embedding_dim,)


def test_mach_udt_embedding():
    model = train_simple_mach_udt()

    embedding = model.embedding_representation({"text": "some sample query"})

    assert embedding.shape == (256,)


def test_mach_udt_decode_params():
    model = train_simple_mach_udt()

    with pytest.raises(
        ValueError,
        match=r"Params must not be 0.",
    ):
        model.set_decode_params(0, 0)

    with pytest.raises(
        ValueError,
        match=r"Cannot eval with top_k_per_eval_aggregation greater than 100.",
    ):
        model.set_decode_params(1, 1000)

    with pytest.raises(
        ValueError,
        match=r"Cannot return more results than the model is trained to predict. Model currently can predict one of 3 classes.",
    ):
        model.set_decode_params(5, 2)

    model.set_decode_params(1, OUTPUT_DIM)

    assert len(model.predict({"text": "something"})) == 1


@pytest.mark.parametrize("integer_target", [True, False])
def test_mach_udt_invalid_class_type(integer_target):
    model = train_simple_mach_udt(integer_target=integer_target)

    label = "1" if integer_target else 1

    with pytest.raises(
        ValueError,
        match=r"Invalid class type. If integer_target=True please use integers as classes, otherwise use strings.",
    ):
        model.get_entity_embedding(label)

    with pytest.raises(
        ValueError,
        match=r"Invalid class type. If integer_target=True please use integers as classes, otherwise use strings.",
    ):
        model.introduce_label([{"text": "something"}], label)


@pytest.mark.parametrize("integer_target", [True, False])
def test_mach_udt_introduce_and_forget(integer_target):
    model = train_simple_mach_udt(integer_target=integer_target)

    label = 4 if integer_target else "4"

    sample = {"text": "something or another with lots of words"}
    assert model.predict(sample)[0][0] != str(label)
    model.introduce_label([sample], label)
    assert model.predict(sample)[0][0] == str(label)
    model.forget(label)
    assert model.predict(sample)[0][0] != str(label)


@pytest.mark.parametrize("integer_target", [True, False])
def test_mach_udt_introduce_existing_class(integer_target):
    model = train_simple_mach_udt(integer_target=integer_target)

    with pytest.raises(
        ValueError,
        match=r"Manually adding a previously seen label: 0. Please use a new label for any new insertions.",
    ):
        model.introduce_label([{"text": "something"}], 0 if integer_target else "0")


@pytest.mark.parametrize("integer_target", [True, False])
def test_mach_udt_forget_non_existing_class(integer_target):
    model = train_simple_mach_udt(integer_target=integer_target)

    with pytest.raises(
        ValueError,
        match=r"Tried to forget label 1000 which does not exist.",
    ):
        model.forget(1000 if integer_target else "1000")


@pytest.mark.parametrize("integer_target", [True, False])
def test_mach_udt_forgetting_everything(integer_target):
    model = train_simple_mach_udt(integer_target=integer_target)

    if integer_target:
        model.forget(0)
        model.forget(1)
        model.forget(2)
    else:
        model.forget("0")
        model.forget("1")
        model.forget("2")

    assert len(model.predict({"text": "something"})) == 0


@pytest.mark.parametrize("integer_target", [True, False])
def test_mach_udt_forgetting_everything_with_clear_index(integer_target):
    model = train_simple_mach_udt(integer_target=integer_target)

    model.clear_index()

    assert len(model.predict({"text": "something"})) == 0


@pytest.mark.parametrize("integer_target", [True, False])
def test_mach_udt_cant_predict_forgotten(integer_target):
    model = train_simple_mach_udt(integer_target=integer_target)

    model.set_decode_params(3, OUTPUT_DIM)
    assert "0" in [class_name for class_name, _ in model.predict({"text": "something"})]
    model.forget(0 if integer_target else "0")
    assert "0" not in [
        class_name for class_name, _ in model.predict({"text": "something"})
    ]


@pytest.mark.parametrize("integer_target", [True, False])
def test_mach_udt_min_num_eval_results_adjusts_on_forget(integer_target):
    model = train_simple_mach_udt(integer_target=integer_target)

    model.set_decode_params(3, OUTPUT_DIM)
    assert len(model.predict({"text": "something"})) == 3
    model.forget(2 if integer_target else "2")
    assert len(model.predict({"text": "something"})) == 2


def test_mach_udt_introduce_document():
    model = train_simple_mach_udt()

    model.introduce_document(
        {"title": "this is a title", "description": "this is a description"},
        strong_column_names=["title"],
        weak_column_names=["description"],
        label="1000",
    )


def test_mach_udt_introduce_documents():
    model = train_simple_mach_udt(integer_target=True)

    new_docs = "NEW_DOCS.csv"
    with open(new_docs, "w") as f:
        f.write("label,title,description\n")
        f.write("4,some title,some description\n")
        f.write("5,some other title,some other description\n")

    model.introduce_documents(
        new_docs,
        strong_column_names=["title"],
        weak_column_names=["description"],
    )

    os.remove(new_docs)


def test_mach_udt_hash_based_methods():
    model = train_simple_mach_udt(integer_target=True)

    hashes = model.predict_hashes({"text": "testing hash based methods"})
    assert len(hashes) == 7

    new_hash_set = set([93, 94, 95, 96, 97, 98, 99])
    assert hashes != new_hash_set

    for _ in range(5):
        model.train_with_hashes(
            [{"text": "testing hash based methods", "label": "93 94 95 96 97 98 99"}],
            learning_rate=0.01,
        )

    new_hashes = model.predict_hashes({"text": "testing hash based methods"})
    assert set(new_hashes) == new_hash_set


@pytest.mark.parametrize("integer_target", [True, False])
def test_mach_save_load_get_set_index(integer_target):
    model = train_simple_mach_udt(integer_target=integer_target)

    make_simple_test_file()
    metrics_before = model.evaluate(SIMPLE_TEST_FILE, metrics=["categorical_accuracy"])

    index = model.get_index()
    save_loc = "index.mach"
    index.save(save_loc)
    if integer_target:
        index = dataset.NumericMachIndex.load(save_loc)
    else:
        index = dataset.StringMachIndex.load(save_loc)

    model.clear_index()
    model.set_index(index)

    metrics_after = model.evaluate(SIMPLE_TEST_FILE, metrics=["categorical_accuracy"])

    assert (
        metrics_before["val_categorical_accuracy"]
        == metrics_after["val_categorical_accuracy"]
    )

    os.remove(save_loc)


@pytest.mark.parametrize("integer_target", [True, False])
def test_mach_setting_wrong_index_type(integer_target):
    model = train_simple_mach_udt(integer_target=integer_target)

    if integer_target:
        index = dataset.StringMachIndex(output_range=OUTPUT_DIM, num_hashes=7)
    else:
        index = dataset.NumericMachIndex({}, output_range=OUTPUT_DIM, num_hashes=7)

    with pytest.raises(
        ValueError,
        match=r"Incorrect index type provided. Index type should be consistent with the integer_target flag.",
    ):
        model.set_index(index)


def test_mach_manual_index_creation():
    model = train_simple_mach_udt(integer_target=True)

    model.set_decode_params(3, OUTPUT_DIM)

    samples = {
        0: "haha one time",
        1: "haha two times",
        2: "haha thrice occurances",
    }

    entity_to_hashes = {
        0: [0, 1, 2, 3, 4, 5, 6],
        1: [7, 8, 9, 10, 11, 12, 13],
        2: [14, 15, 16, 17, 18, 19, 20],
    }

    index = dataset.NumericMachIndex(
        entity_to_hashes=entity_to_hashes,
        output_range=OUTPUT_DIM,
        num_hashes=7,
    )

    model.set_index(index)

    make_simple_test_file()
    model.train(SIMPLE_TEST_FILE, learning_rate=0.01, epochs=10)

    for label, sample in samples.items():
        new_hashes = model.predict_hashes({"text": sample})
        assert set(new_hashes) == set(entity_to_hashes[label])
