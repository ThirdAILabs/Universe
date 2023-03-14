import pytest
from download_dataset_fixtures import download_scifact_dataset
from thirdai import bolt

pytestmark = [pytest.mark.unit]


SIMPLE_TEST_FILE = "mach_udt_test.csv"


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
    all_recommended_documents = [
        doc_score[0] for doc_score in model.evaluate(filename=supervised_tst)
    ]

    all_relevant_documents = get_relevant_documents(supervised_tst)

    precision = calculate_precision(all_relevant_documents, all_recommended_documents)

    assert precision > 0.5

    return precision


def make_simple_train_data(invalid=False):
    with open(SIMPLE_TEST_FILE, "w") as f:
        f.write("text,label\n")
        f.write("haha,0\n")
        f.write("haha,1\n")
        if invalid:
            f.write("haha,2\n")


def train_simple_mach_udt(integer_target):
    model = bolt.UniversalDeepTransformer(
        data_types={
            "text": bolt.types.text(contextual_encoding="local"),
            "label": bolt.types.categorical(),
        },
        target="label",
        n_target_classes=2,
        integer_target=integer_target,
        options={"extreme_classification": True},
    )

    model.train(SIMPLE_TEST_FILE, epochs=1, learning_rate=0.001)

    return model


def test_mach_udt_on_scifact(download_scifact_dataset):
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
        integer_target=True,
        options={"extreme_classification": True},
    )

    metrics = model.cold_start(
        filename=unsupervised_file,
        strong_column_names=["TITLE"],
        weak_column_names=["TEXT"],
        learning_rate=0.001,
        epochs=10,
        metrics=[
            "precision@1",
            "recall@10",
        ],
    )

    assert metrics["precision@1"][-1] > 0.95

    metrics = model.train(
        filename=supervised_trn,
        learning_rate=0.001,
        epochs=10,
        metrics=[
            "precision@1",
            "recall@10",
        ],
    )

    assert metrics["precision@1"][-1] > 0.95

    before_save_precision = evaluate_model(model, supervised_tst)

    save_loc = "model.bolt"
    model.save(save_loc)
    model = bolt.UniversalDeepTransformer.load(save_loc)

    after_save_precision = evaluate_model(model, supervised_tst)

    assert before_save_precision == after_save_precision


def test_mach_udt_string_target():
    pass


def test_mach_udt_string_target_too_many_classes():
    make_simple_train_data(invalid=True)

    with pytest.raises(
        ValueError,
        match=r"Unable to find column with name 'SOME RANDOM NAME'.",
    ):
        train_simple_mach_udt(integer_target=False)


def test_mach_udt_integer_target_label_too_large():
    make_simple_train_data(invalid=True)

    with pytest.raises(
        ValueError,
        match=r"Unable to find column with name 'SOME RANDOM NAME'.",
    ):
        train_simple_mach_udt(integer_target=True)


def test_mach_udt_entity_embedding():
    make_simple_train_data(invalid=True)

    model = train_simple_mach_udt(integer_target=False)

    model.get_entity_embedding()


def test_mach_udt_embedding():
    make_simple_train_data(invalid=True)

    model = train_simple_mach_udt(integer_target=False)

    model.embedding_representation()


# test the decoding -> c++ test?
