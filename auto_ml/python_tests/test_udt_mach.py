import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import pytest
from download_dataset_fixtures import download_scifact_dataset
from thirdai import bolt, dataset

pytestmark = [pytest.mark.unit]


SIMPLE_TEST_FILE = "mach_udt_test.csv"
OUTPUT_DIM = 100
NUM_HASHES = 7


def make_simple_test_file(invalid_data=False):
    with open(SIMPLE_TEST_FILE, "w") as f:
        f.write("text,label\n")
        f.write("haha one time,0\n")
        f.write("haha two times,1\n")
        f.write("haha thrice occurrences,2\n")
        if invalid_data:
            f.write("haha,3\n")


def train_simple_mach_udt(
    invalid_data=False,
    embedding_dim=256,
    use_bias=True,
    rlhf_args={},
    mach_sampling_threshold=0.2,
    output_dim=OUTPUT_DIM,
):
    make_simple_test_file(invalid_data=invalid_data)

    model = bolt.UniversalDeepTransformer(
        data_types={
            "text": bolt.types.text(contextual_encoding="local"),
            "label": bolt.types.categorical(),
        },
        target="label",
        n_target_classes=3,
        integer_target=True,
        options={
            "extreme_classification": True,
            "embedding_dimension": embedding_dim,
            "extreme_output_dim": output_dim,
            "hidden_bias": use_bias,
            "output_bias": use_bias,
            **rlhf_args,
            "mach_sampling_threshold": mach_sampling_threshold,
        },
    )

    model.train(
        SIMPLE_TEST_FILE, epochs=5, learning_rate=0.001, shuffle_reservoir_size=32000
    )

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


def scifact_model(n_target_classes):
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
    return model


def train_on_scifact(download_scifact_dataset, coldstart):
    (
        unsupervised_file,
        supervised_trn,
        supervised_tst,
        n_target_classes,
    ) = download_scifact_dataset

    model = scifact_model(n_target_classes=n_target_classes)

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


@pytest.fixture(scope="session")
def train_mach_on_scifact_with_cold_start(download_scifact_dataset):
    return train_on_scifact(download_scifact_dataset, coldstart=True)


def test_mach_udt_on_scifact(train_mach_on_scifact_with_cold_start):
    _, metrics, _ = train_mach_on_scifact_with_cold_start

    assert metrics["train_precision@1"][-1] > 0.45


def test_mach_udt_on_scifact_save_load(train_mach_on_scifact_with_cold_start):
    model, _, supervised_tst = train_mach_on_scifact_with_cold_start

    before_save_precision = evaluate_model(model, supervised_tst)

    assert before_save_precision > 0.45

    save_loc = "model.bolt"
    model.save(save_loc)
    model = bolt.UniversalDeepTransformer.load(save_loc)

    after_save_precision = evaluate_model(model, supervised_tst)

    assert before_save_precision == after_save_precision

    os.remove(save_loc)


def test_mach_udt_on_scifact_model_porting(
    train_mach_on_scifact_with_cold_start, download_scifact_dataset
):
    _, _, _, n_classes = download_scifact_dataset
    model, _, supervised_tst = train_mach_on_scifact_with_cold_start

    before_porting_precision = evaluate_model(model, supervised_tst)

    new_model = scifact_model(n_target_classes=n_classes)

    new_bolt_model = bolt.nn.Model.from_params(model._get_model().params())
    new_model._set_model(new_bolt_model)

    new_model.set_index(model.get_index())

    # Check that the accuracy matches in the ported model.
    assert before_porting_precision == evaluate_model(new_model, supervised_tst)

    # Check that predictions match in the ported model.
    test_df = pd.read_csv(supervised_tst)
    batch = [{"QUERY": text} for text in test_df["QUERY"]]

    assert model.predict_batch(batch) == new_model.predict_batch(batch)


def test_mach_udt_label_too_large():
    with pytest.raises(
        ValueError,
        match=r"Invalid entity in index: 3.",
    ):
        train_simple_mach_udt(invalid_data=True)


@pytest.mark.parametrize("embedding_dim", [128, 256])
def test_mach_udt_entity_embedding(embedding_dim):
    model = train_simple_mach_udt(embedding_dim=embedding_dim)
    output_labels = [0, 1]
    for output_id, output_label in enumerate(output_labels):
        embedding = model.get_entity_embedding(output_label)
        assert embedding.shape == (embedding_dim,)


def test_mach_udt_embedding():
    model = train_simple_mach_udt()

    embedding = model.embedding_representation([{"text": "some sample query"}])

    assert embedding.shape == (256,)

    embedding = model.embedding_representation(
        [{"text": "some sample query"}, {"text": "some sample query"}]
    )

    assert embedding.shape == (2, 256)


def test_mach_udt_decode_params():
    model = train_simple_mach_udt()

    with pytest.raises(
        ValueError,
        match=r"Params must not be 0.",
    ):
        model.set_decode_params(0, 0)

    with pytest.raises(
        ValueError,
        match=r"Cannot eval with num_buckets_to_eval greater than 100.",
    ):
        model.set_decode_params(1, 1000)

    with pytest.raises(
        ValueError,
        match=r"Cannot return more results than the model is trained to predict. Model currently can predict one of 3 classes.",
    ):
        model.set_decode_params(5, 2)

    model.set_decode_params(1, OUTPUT_DIM)

    assert len(model.predict({"text": "something"})) == 1


def test_mach_udt_topk_predict():
    model = train_simple_mach_udt()

    model.set_decode_params(1, 5)

    assert len(model.predict({"text": "something"})) == 1

    assert len(model.predict({"text": "something"}, top_k=2)) == 2

    with pytest.raises(
        ValueError,
        match=r"Cannot return more results than the model is trained to predict. Model currently can predict one of 3 classes.",
    ):
        model.predict({"text": "something"}, top_k=4)


def test_mach_udt_introduce_and_forget():
    model = train_simple_mach_udt()

    label = 4

    sample = {"text": "something or another with lots of words"}
    assert model.predict(sample)[0][0] != label
    model.introduce_label([sample], label)
    assert model.predict(sample)[0][0] == label
    model.forget(label)
    assert model.predict(sample)[0][0] != label


def test_mach_udt_introduce_existing_class():
    model = train_simple_mach_udt()

    with pytest.raises(
        ValueError,
        match=r"Manually adding a previously seen label: 0. Please use a new label for any new insertions.",
    ):
        model.introduce_label([{"text": "something"}], 0)


def test_mach_udt_forget_non_existing_class():
    model = train_simple_mach_udt()

    with pytest.raises(
        ValueError,
        match=r"Invalid entity in index: 1000.",
    ):
        model.forget(1000)


def test_mach_udt_forgetting_everything():
    model = train_simple_mach_udt()

    model.forget(0)
    model.forget(1)
    model.forget(2)

    assert len(model.predict({"text": "something"})) == 0


def test_mach_udt_forgetting_everything_with_clear_index():
    model = train_simple_mach_udt()

    model.clear_index()

    assert len(model.predict({"text": "something"})) == 0


def test_mach_udt_cant_predict_forgotten():
    model = train_simple_mach_udt()

    model.set_decode_params(3, OUTPUT_DIM)
    assert 0 in [class_name for class_name, _ in model.predict({"text": "something"})]
    model.forget(0)
    assert 0 not in [
        class_name for class_name, _ in model.predict({"text": "something"})
    ]


def test_mach_udt_min_num_eval_results_adjusts_on_forget():
    model = train_simple_mach_udt()

    model.set_decode_params(3, OUTPUT_DIM)
    assert len(model.predict({"text": "something"})) == 3
    model.forget(2)
    assert len(model.predict({"text": "something"})) == 2


def test_mach_udt_introduce_document():
    model = train_simple_mach_udt()

    model.introduce_document(
        {"title": "this is a title", "description": "this is a description"},
        strong_column_names=["title"],
        weak_column_names=["description"],
        label=1000,
    )


@pytest.mark.parametrize("fast_approximation", [True, False])
def test_mach_udt_introduce_documents(fast_approximation):
    model = train_simple_mach_udt()

    new_docs = "NEW_DOCS.csv"
    with open(new_docs, "w") as f:
        f.write("label,title,description\n")
        f.write("4,some title,some description\n")
        f.write("5,some other title,some other description\n")

    model.introduce_documents(
        new_docs,
        strong_column_names=["title"],
        weak_column_names=["description"],
        fast_approximation=fast_approximation,
    )

    os.remove(new_docs)


def test_mach_udt_hash_based_methods():
    # Set mach_sampling_threshold = 1.0 to ensure that we use MACH index for
    # active neuron selection.
    model = train_simple_mach_udt(mach_sampling_threshold=1.0)

    hashes = model.predict_hashes(
        {"text": "testing hash based methods"},
        sparse_inference=False,
        force_non_empty=False,
    )
    assert len(hashes) == 7

    # All hashes in new_hash_set represent non-empty buckets since they are
    # the hashes of an entity. This is important since we're using MACH index
    # for active neuron selection.
    model.introduce_label([{"text": "text that will map to different buckets"}], 1000)
    new_hash_set = set(model.get_index().get_entity_hashes(1000))
    assert hashes != new_hash_set

    for _ in range(5):
        model.train_with_hashes(
            [
                {
                    "text": "testing hash based methods",
                    "label": " ".join(map(str, new_hash_set)),
                }
            ],
            learning_rate=0.01,
        )

    new_hashes = model.predict_hashes({"text": "testing hash based methods"})
    assert set(new_hashes) == new_hash_set

    # Now set mach_sampling_threshold = 0.0 to ensure that we use LSH index for
    # active neuron selection.
    model = train_simple_mach_udt(mach_sampling_threshold=0.0)

    hashes = model.predict_hashes({"text": "testing hash based methods"})
    assert len(hashes) == 7

    # Hashes are empty buckets. This is fine since we are using LSH index for
    # active neuron selection.
    empty_hashes = [
        i for i in range(100) if len(model.get_index().get_hash_to_entities(i)) == 0
    ]
    new_hash_set = set(empty_hashes[:7])
    assert hashes != new_hash_set

    for _ in range(10):
        model.train_with_hashes(
            [
                {
                    "text": "testing hash based methods",
                    "label": " ".join(map(str, new_hash_set)),
                }
            ],
            learning_rate=0.01,
        )

    new_hashes = model.predict_hashes(
        {"text": "testing hash based methods"},
        sparse_inference=False,
        force_non_empty=False,
    )
    assert set(new_hashes) == new_hash_set


def test_mach_output_correctness():
    model = train_simple_mach_udt(output_dim=50)

    # Suppose the label corresponding to the given text is 2.
    predicted_hashes = model.predict_hashes(
        {"text": "testing output correctness"},
        force_non_empty=True,
    )

    mach_index = model.get_index()

    original_hashes = mach_index.get_entity_hashes(2)

    expected_ratio = len(set(predicted_hashes) & set(original_hashes)) / len(
        original_hashes
    )

    num_correct_buckets = model.output_correctness(
        [{"text": "testing output correctness"}], labels=[2]
    )[0]

    current_ratio = num_correct_buckets / (mach_index.num_hashes())

    assert expected_ratio == current_ratio


def test_mach_save_load_get_set_index():
    model = train_simple_mach_udt()
    metrics = ["recall@5", "precision@5"]

    make_simple_test_file()
    metrics_before = model.evaluate(SIMPLE_TEST_FILE, metrics=metrics)

    index = model.get_index()
    save_loc = "index.mach"
    index.save(save_loc)
    index = dataset.MachIndex.load(save_loc)

    model.clear_index()
    model.set_index(index)

    metrics_after = model.evaluate(SIMPLE_TEST_FILE, metrics=metrics)

    assert metrics_before == metrics_after

    os.remove(save_loc)


def test_mach_manual_index_creation():
    model = train_simple_mach_udt()

    model.set_decode_params(3, OUTPUT_DIM)

    samples = {
        0: "haha one time",
        1: "haha two times",
        2: "haha thrice occurrences",
    }

    entity_to_hashes = {
        0: [0, 1, 2, 3, 4, 5, 6],
        1: [7, 8, 9, 10, 11, 12, 13],
        2: [14, 15, 16, 17, 18, 19, 20],
    }

    index = dataset.MachIndex(
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


def test_mach_without_bias():
    model = train_simple_mach_udt(use_bias=False)

    bolt_model = model._get_model()

    ops = bolt_model.ops()

    hidden_layer = ops[0]  # hidden layer
    output_layer = ops[1]  # output layer

    assert np.all(hidden_layer.biases == 0)
    assert np.all(output_layer.biases == 0)


def test_load_balancing():
    model = train_simple_mach_udt()
    num_hashes = 8
    half_num_hashes = 4
    sample = {"text": "tomato"}

    # Set the index so that we know that the number of hashes is 8.
    model.set_index(
        dataset.MachIndex({}, output_range=OUTPUT_DIM, num_hashes=num_hashes)
    )

    # This gives the top 8 locations where the new sample will end up.
    hash_locs = model.predict_hashes(sample, force_non_empty=False)

    # Create a new index with 4 hashes, with elements to 4 of the 8 top locations
    # for the new element.
    new_index = dataset.MachIndex(
        {i: [h] * half_num_hashes for i, h in enumerate(hash_locs[:half_num_hashes])},
        output_range=OUTPUT_DIM,
        num_hashes=half_num_hashes,
    )
    model.set_index(new_index)

    # Insert an id for the same sample without load balancing to ensure that
    # it goes to different locations than with load balancing
    label_without_load_balancing = 9999
    model.introduce_label(
        input_batch=[sample],
        label=label_without_load_balancing,
    )

    # We are sampling 8 locations, this should be the top 8 locations we determined
    # earlier. However since we have inserted elements in the index in 4 of these
    # top 8 locations it should insert the new element in the other 4 locations
    # due to the load balancing constraint.
    label_with_load_balancing = 10000
    model.introduce_label(
        input_batch=[sample],
        label=label_with_load_balancing,
        num_buckets_to_sample=num_hashes,
    )

    hashes_with_load_balancing = model.get_index().get_entity_hashes(
        label_with_load_balancing
    )
    hashes_without_load_balancing = model.get_index().get_entity_hashes(
        label_without_load_balancing
    )

    # Check that it inserts into the empty buckets without load balancing.
    assert set(hashes_with_load_balancing) == set(hash_locs[half_num_hashes:])

    # Check that the buckets it inserts into with load balancing is different
    # than the buckets it inserts into without load balancing
    assert set(hashes_with_load_balancing) != set(hashes_without_load_balancing)


def test_mach_sparse_inference():
    """
    This test checks that if we create a mach index that with a number of non
    empty buckets that puts it under the theshold for mach index sampling, only the
    non empty buckets are returned by sparse inference. It then checks that the
    returned buckets are updated as the index is modified, and then finally
    checks that it no longer uses mach sampling after the index sufficient non
    empty buckets.
    """
    model = train_simple_mach_udt()

    model.clear_index()

    model.set_index(
        dataset.MachIndex(
            {1: [10], 2: [20], 3: [30]}, output_range=OUTPUT_DIM, num_hashes=1
        )
    )

    input_vec = bolt.nn.Tensor(dataset.make_sparse_vector([0], [1.0]), 100_000)

    output = model._get_model().forward([input_vec], use_sparsity=True)[0]
    assert set(output.active_neurons[0]) == set([10, 20, 30])

    model.set_index(
        dataset.MachIndex(
            {1: [10], 2: [20], 3: [30], 4: [40]},
            output_range=OUTPUT_DIM,
            num_hashes=1,
        )
    )

    output = model._get_model().forward([input_vec], use_sparsity=True)[0]
    assert set(output.active_neurons[0]) == set([10, 20, 30, 40])

    model.forget(label=3)

    output = model._get_model().forward([input_vec], use_sparsity=True)[0]
    assert set(output.active_neurons[0]) == set([10, 20, 40])

    # This is above the threshold for mach index sampling, so it should revert back to LSH
    model.set_index(
        dataset.MachIndex(
            {i * 10: [i] for i in range(OUTPUT_DIM // 2)},
            output_range=OUTPUT_DIM,
            num_hashes=1,
        )
    )

    # When we set an index with 50% sparsity it will autotune the sampling, it
    # will decide to not use any sort of sampling for this level of sparsity and
    # so the output should be dense.
    output = model._get_model().forward([input_vec], use_sparsity=True)[0]
    assert output.active_neurons == None
    assert output.activations.shape == (1, OUTPUT_DIM)


def test_associate():
    model = train_simple_mach_udt(
        rlhf_args={
            "rlhf": True,
            "rlhf_balancing_docs": 100,
            "rlhf_balancing_samples_per_doc": 10,
        }
    )

    target_sample = {"text": "random sample text"}
    model.introduce_label([target_sample], label=200)
    target_hashes = set(model.predict_hashes(target_sample))

    different_hashes = list(set(range(OUTPUT_DIM)).difference(target_hashes))
    different_hashes = random.choices(different_hashes, k=7)
    different_hashes = " ".join([str(x) for x in different_hashes])

    source_sample = {"text": "tomato", "label": different_hashes}
    target_sample["label"] = " ".join([str(x) for x in target_hashes])
    for _ in range(100):
        model.train_with_hashes([source_sample, target_sample], 0.001)
    del source_sample["label"]

    target_hashes = set(model.predict_hashes(target_sample))

    model.introduce_label([source_sample], label=100)
    source_hashes = set(model.predict_hashes(source_sample))

    original_intersection = len(target_hashes.intersection(source_hashes))

    for _ in range(100):
        model.associate(
            [(source_sample["text"], target_sample["text"], 1.0)], n_buckets=7
        )

    new_target_hashes = set(model.predict_hashes(target_sample))
    new_source_hashes = set(model.predict_hashes(source_sample))

    new_intersection = len(new_target_hashes.intersection(new_source_hashes))

    assert new_intersection > original_intersection


def test_enable_rlhf():
    model = train_simple_mach_udt()

    with pytest.raises(
        RuntimeError,
        match=r"This model was not configured to support rlhf. Please pass {'rlhf': True} in the model options or call enable_rlhf().",
    ):
        model.associate([("text", "text"), 1.0], n_buckets=7)

    model.enable_rlhf(num_balancing_docs=100, num_balancing_samples_per_doc=10)

    make_simple_test_file()

    model.train(
        SIMPLE_TEST_FILE, epochs=5, learning_rate=0.001, shuffle_reservoir_size=32000
    )

    model.associate([("text", "text"), 1.0], n_buckets=7)


def regularized_introduce_helper(model, num_random_hashes):
    """Returns an array counting the number of hashes in each bucket after
    introducing three identical samples"""

    for label in range(3):
        model.introduce_label(
            [{"text": "some text"}],
            label,
            num_buckets_to_sample=None,
            num_random_hashes=num_random_hashes,
        )

    index = model.get_index()
    load = np.zeros(OUTPUT_DIM, dtype=np.int32)
    for i in range(len(load)):
        load[i] = len(index.get_hash_to_entities(i))

    return load


def test_introduce_hash_regularization():
    model = train_simple_mach_udt()

    model.clear_index()

    # without any regularization or balancing, introducing 3 labels with the
    # same representative sample should yield 3 sets of identical hashes
    load = regularized_introduce_helper(model, num_random_hashes=0)
    assert np.sum(load > 0) == NUM_HASHES

    model.clear_index()

    # when 2 of the 7 hashes in every new doc are random there should be more
    # than NUM_HASHES non-zeroes in the index's load
    load = regularized_introduce_helper(model, num_random_hashes=2)
    assert np.sum(load > 0) > NUM_HASHES


def test_udt_mach_train_batch():
    model = train_simple_mach_udt()

    model.train_batch([{"text": "some text", "label": "2"}], learning_rate=0.001)


def test_udt_mach_num_buckets_to_sample_and_switching_index_num_hashes():
    model = train_simple_mach_udt()

    with pytest.raises(
        ValueError,
        match=r"Sampling from fewer buckets than num_hashes is not supported. If you'd like to introduce using fewer hashes, please reset the number of hashes for the index.",
    ):
        model.introduce_label(
            [{"text": "some text"}], 0, num_buckets_to_sample=NUM_HASHES - 1
        )

    new_index = dataset.MachIndex(num_hashes=NUM_HASHES - 1, output_range=OUTPUT_DIM)

    model.set_index(new_index)

    model.introduce_label(
        [{"text": "some text"}], 0, num_buckets_to_sample=NUM_HASHES - 1
    )


def test_udt_mach_fast_approximation_handles_commas():
    model = train_simple_mach_udt()
    model.clear_index()

    with open("temp.csv", "w") as out:
        out.write("strong,weak,label\n")
        out.write('"a string, with, commas","another, one",0\n')

    # We only care that it doesn't throw an error
    model.introduce_documents(
        "temp.csv",
        strong_column_names=["strong"],
        weak_column_names=["weak"],
        fast_approximation=True,
    )

    os.remove("temp.csv")
