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
    output = model.evaluate(filename=supervised_tst)

    all_recommended_documents = []
    for sample in output:
        all_recommended_documents.append([int(doc) for doc, score in sample])

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


def train_simple_mach_udt(integer_target, embedding_dim=256):
    model = bolt.UniversalDeepTransformer(
        data_types={
            "text": bolt.types.text(contextual_encoding="local"),
            "label": bolt.types.categorical(),
        },
        target="label",
        n_target_classes=2,
        integer_target=integer_target,
        options={"extreme_classification": True, "embedding_dimension": embedding_dim},
    )

    model.train(SIMPLE_TEST_FILE, epochs=1, learning_rate=0.001)

    return model


class SupervisedTrainCallback(bolt.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.prev_metric = 0

    def on_epoch_end(self, model, train_state):
        cur_metric = train_state.get_validation_metric_values("precision@1")[-1]
        if cur_metric < self.prev_metric:
            train_state.stop_training = True
        else:
            self.prev_metric = cur_metric


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
        options={"extreme_classification": True, "embedding_dimension": 1024},
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

    validation = bolt.Validation(
        supervised_tst,
        metrics=["precision@1"],
    )

    # assert metrics["precision@1"][-1] > 0.95

    metrics = model.train(
        filename=supervised_trn,
        learning_rate=0.001,
        epochs=10,
        metrics=[
            "precision@1",
            "recall@10",
        ],
        validation=validation,
        callbacks=[SupervisedTrainCallback()],
    )

    # assert metrics["precision@1"][-1] > 0.95

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
        match=r"Received additional category*",
    ):
        train_simple_mach_udt(integer_target=False)


def test_mach_udt_integer_target_label_too_large():
    make_simple_train_data(invalid=True)

    with pytest.raises(
        ValueError,
        match=r"Received label 2 larger than or equal to n_target_classes.",
    ):
        train_simple_mach_udt(integer_target=True)


@pytest.mark.unit
@pytest.mark.parametrize(
    "embedding_dim, integer_label",
    [(128, True), (128, False), (256, True), (256, False)],
)
def test_mach_udt_entity_embedding(embedding_dim, integer_label):
    make_simple_train_data()
    model = train_simple_mach_udt(
        integer_target=integer_label, embedding_dim=embedding_dim
    )
    output_labels = [0, 1] if integer_label else ["0", "1"]
    for output_id, output_label in enumerate(output_labels):
        embedding = model.get_entity_embedding(output_label)
        assert embedding.shape == (embedding_dim,)


def test_mach_udt_embedding():
    make_simple_train_data(invalid=False)

    model = train_simple_mach_udt(integer_target=False)

    embedding = model.embedding_representation({"QUERY": "some sample query"})

    assert embedding.shape == (256,)
