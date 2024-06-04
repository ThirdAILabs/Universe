import os
import platform
import textwrap

import pytest
from thirdai import bolt

pytestmark = [pytest.mark.unit]

TRAIN_FILE = "tempTrainFile.csv"
TEST_FILE = "tempTestFile.csv"
METADATA_FILE = "tempMetaFile.csv"


def write_lines_to_file(file, lines):
    with open(file, "w") as f:
        for line in lines:
            f.writelines(line + "\n")


def make_simple_trained_model(
    embedding_dim=None,
    integer_label=False,
    text_encoding_type="none",
    numerical_temporal=True,
):
    write_lines_to_file(
        TRAIN_FILE,
        [
            "userId,movieId,timestamp,hoursWatched,genres,description",
            "0,0,2022-08-29,2,fiction-comedy-drama,a movie",
            "1,0,2022-08-30,2,fiction-romance,a movie",
            "1,1,2022-08-31,1,romance-comedy,a movie",
            # if integer_label = false, we build a model that accepts
            # arbitrary string labels; the model does not expect integer
            # labels in the range [0, n_labels - 1]. We test this by
            # checking that the model does not throw an error when given
            # a label outside of this range. Since n_labels = 3, we set
            # movieId = 4 in the last sample and expect that the model
            # trains just fine.
            (
                "1,2,2022-09-01,3,fiction-comedy,a movie"
                if integer_label
                else "1,4,2022-09-01,3,fiction-comedy,a movie"
            ),
        ],
    )

    write_lines_to_file(
        TEST_FILE,
        [
            "userId,movieId,timestamp,hoursWatched,genres,description",
            "0,1,2022-10-31,5,fiction-drama,a movie",
            # See above comment about the last line of the mock train file.
            (
                "1,0,2022-11-01,0.5,fiction-comedy,a movie"
                if integer_label
                else "4,0,2022-11-01,0.5,fiction-comedy,a movie"
            ),
        ],
    )

    model = bolt.UniversalDeepTransformer(
        data_types={
            "userId": bolt.types.categorical(),
            "movieId": bolt.types.categorical(
                n_classes=3, type="int" if integer_label else "str"
            ),
            "timestamp": bolt.types.date(),
            "hoursWatched": bolt.types.numerical(range=(0, 5)),
            "genres": bolt.types.categorical(delimiter="-"),
            "description": bolt.types.text(contextual_encoding=text_encoding_type),
        },
        temporal_tracking_relationships={
            "userId": ["movieId"],
            **({"userId": ["hoursWatched"]} if numerical_temporal else {}),
        },
        target="movieId",
        **({"embedding_dimension": embedding_dim} if embedding_dim else {}),
    )

    model.train(TRAIN_FILE, epochs=2, learning_rate=0.01, batch_size=2048)

    return model


def single_sample():
    return {
        "userId": "0",
        "timestamp": "2022-12-20",
        "hoursWatched": "1",
        "genres": "fiction-drama",
        "description": "",
    }


def batch_sample():
    return [single_sample(), single_sample(), single_sample()]


def single_update():
    return {
        "userId": "0",
        "movieId": "1",
        "timestamp": "2022-12-20",
        "hoursWatched": "1",
        "genres": "fiction-drama",
        "description": "",
    }


def batch_update():
    return [single_update(), single_update(), single_update()]


def compare_explanations(explanations_1, explanations_2, assert_mode):
    all_equal = len(explanations_1) == len(explanations_2) and all(
        exp_1 == exp_2 for exp_1, exp_2 in zip(explanations_1, explanations_2)
    )

    # If we want to assert equality, we want everything to be equal
    # Otherwise, we want something to be inequal.
    assert_equal = assert_mode == "equal"
    assert assert_equal == all_equal


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="Throwing an exception leads to an access violation on windows.",
)
@pytest.mark.release
def test_temporal_not_in_data_type_throws():
    with pytest.raises(
        ValueError,
        match=r"Tracking key column 'user' is not specified in data_types.",
    ):
        bolt.UniversalDeepTransformer(
            data_types={
                "date": bolt.types.date(),
                "item": bolt.types.categorical(n_classes=3),
            },
            temporal_tracking_relationships={"user": ["item"]},
            target="item",
        )

    with pytest.raises(
        ValueError,
        match=r"The tracked column 'other_item' is not found in data_types.",
    ):
        bolt.UniversalDeepTransformer(
            data_types={
                "date": bolt.types.date(),
                "user": bolt.types.categorical(),
                "item": bolt.types.categorical(n_classes=3),
            },
            temporal_tracking_relationships={"user": ["other_item"]},
            target="item",
        )


@pytest.mark.release
def test_save_load():
    save_file = "savefile.bolt"
    model = make_simple_trained_model(integer_label=False)
    model.save(save_file)
    saved_model = bolt.UniversalDeepTransformer.load(filename=save_file)

    eval_res = model.evaluate(TEST_FILE)
    del eval_res["val_times"]
    saved_eval_res = saved_model.evaluate(TEST_FILE)
    del saved_eval_res["val_times"]
    assert eval_res == saved_eval_res

    model.index(single_update())
    saved_model.index(single_update())

    predict_res = model.predict(single_sample())
    saved_predict_res = saved_model.predict(single_sample())
    assert (predict_res == saved_predict_res).all()

    predict_batch_res = model.predict_batch(batch_sample())
    saved_predict_batch_res = saved_model.predict_batch(batch_sample())
    assert (predict_batch_res == saved_predict_batch_res).all()


@pytest.mark.release
def test_multiple_predict_returns_same_results():
    model = make_simple_trained_model(integer_label=False)
    first = model.predict(single_sample())
    second = model.predict(single_sample())
    assert (first == second).all()

    first_batch = model.predict_batch(batch_sample())
    second_batch = model.predict_batch(batch_sample())
    assert (first_batch == second_batch).all()


@pytest.mark.release
def test_index_changes_predict_result():
    model = make_simple_trained_model(integer_label=False, numerical_temporal=False)
    first = model.predict(single_sample())
    model.index(single_update())
    second = model.predict(single_sample())
    assert (first != second).any()

    model.index_batch(batch_update())
    third = model.predict(single_sample())
    assert (second != third).any()


@pytest.mark.release
def test_embedding_representation_returns_correct_dimension():
    for embedding_dim in [128, 256]:
        model = make_simple_trained_model(embedding_dim=embedding_dim)
        embedding = model.embedding_representation([single_sample()])
        assert embedding.shape == (embedding_dim,)
        assert (embedding != 0).any()


@pytest.mark.unit
@pytest.mark.parametrize(
    "embedding_dim, integer_label",
    [(128, True), (128, False), (256, True), (256, False)],
)
def test_entity_embedding(embedding_dim, integer_label):
    model = make_simple_trained_model(
        embedding_dim=embedding_dim, integer_label=integer_label
    )

    if integer_label:
        output_labels = [0, 1, 2]
        labels_to_neurons = output_labels
    else:
        output_labels = ["0", "1", "4"]
        labels_to_neurons = {model.class_name(n): n for n in range(3)}

    for output_label in output_labels:
        embedding = model.get_entity_embedding(output_label)
        assert embedding.shape == (embedding_dim,)
        weights = model._get_model().ops()[1].weights

        assert (weights[labels_to_neurons[output_label]] == embedding).all()


@pytest.mark.release
def test_entity_embedding_fails_on_large_label():
    model = make_simple_trained_model(embedding_dim=100, integer_label=True)

    with pytest.raises(
        ValueError,
        match=r"Passed in neuron_id too large for this layer. Should be less than the output dim of 3.",
    ):
        embedding = model.get_entity_embedding(100000)


@pytest.mark.release
def test_explanations_total_percentage():
    model = make_simple_trained_model(integer_label=False, numerical_temporal=False)
    explanations = model.explain(single_sample())
    total_percentage = 0
    for _, percent in explanations:
        total_percentage += abs(percent)

    assert total_percentage > 0.99 and total_percentage < 1.01


@pytest.mark.release
def test_different_explanation_target_returns_different_results():
    model = make_simple_trained_model(integer_label=False, numerical_temporal=False)

    explain_target_1 = model.explain(single_sample(), target_class="1")
    explain_target_2 = model.explain(single_sample(), target_class="4")
    compare_explanations(explain_target_1, explain_target_2, assert_mode="not_equal")


def test_explanations_target_label_format():
    model = make_simple_trained_model(integer_label=False, numerical_temporal=False)
    # Call this method to make sure it does not throw an error
    model.explain(single_sample(), target_class="1")
    with pytest.raises(ValueError, match=r"Received an integer but*"):
        model.explain(single_sample(), target_class=1)

    model = make_simple_trained_model(integer_label=True, numerical_temporal=False)
    # Call this method to make sure it does not throw an error
    model.explain(single_sample(), target_class=1)
    with pytest.raises(ValueError, match=r"Received a string but*"):
        model.explain(single_sample(), target_class="1")


@pytest.mark.release
def test_neuron_id_to_target_class_map():
    model = make_simple_trained_model(integer_label=False)
    prediction = model.predict(single_sample())
    n_output_neurons = len(prediction)

    # "0", "1", and "4" are the three possible labels in the
    # mock train / eval dataset when integer_label = False
    labels_seen = {"0": False, "1": False, "4": False}

    for neuron_id in range(n_output_neurons):
        label = model.class_name(neuron_id)
        labels_seen[label] = True

    assert all([seen for seen in labels_seen.values()])


@pytest.mark.release
def test_reset_clears_history():
    model = make_simple_trained_model(integer_label=False)
    model.reset_temporal_trackers()
    first = model.predict(single_sample())

    model.index(single_update())
    after_index = model.predict(single_sample())
    assert (first != after_index).any()

    model.reset_temporal_trackers()
    after_reset = model.predict(single_sample())
    assert (first == after_reset).all()


@pytest.mark.release
def test_works_without_temporal_relationships():
    write_lines_to_file(
        TRAIN_FILE,
        [
            "userId,movieId,hoursWatched,genres",
            "0,0,2,fiction-drama",
            "1,0,3,fiction-comedy",
        ],
    )

    write_lines_to_file(
        TEST_FILE,
        [
            "userId,movieId,hoursWatched,genres",
            "0,1,5,fiction-drama",
            "2,0,0.5,fiction-comedy",
        ],
    )

    model = bolt.UniversalDeepTransformer(
        data_types={
            "userId": bolt.types.categorical(),
            "movieId": bolt.types.categorical(n_classes=3),
            "hoursWatched": bolt.types.numerical(range=(0, 5)),
            "genres": bolt.types.categorical(delimiter="-"),
        },
        target="movieId",
    )

    model.train(TRAIN_FILE, epochs=2, learning_rate=0.01, batch_size=2048)
    model.evaluate(TEST_FILE)

    # No assertion as we just want to know that there is no error.


def test_return_train_metrics():
    model = make_simple_trained_model()

    metrics = model.train(TEST_FILE, epochs=1, metrics=["categorical_accuracy"])
    assert metrics["train_categorical_accuracy"][-1] >= 0


def test_return_train_metrics_streamed():
    model = make_simple_trained_model()
    model.reset_temporal_trackers()

    batch_size = 1
    max_in_memory_batches = 1

    with open(TRAIN_FILE, "r") as f:
        num_samples = len(f.readlines()) - 1

    metrics = model.train(
        TRAIN_FILE,
        epochs=1,
        metrics=["categorical_accuracy"],
        max_in_memory_batches=batch_size,
        batch_size=max_in_memory_batches,
    )

    assert metrics["train_categorical_accuracy"][-1] >= 0
    assert len(metrics["train_categorical_accuracy"]) == num_samples / batch_size


def test_udt_override_input_dim():
    udt_model = bolt.UniversalDeepTransformer(
        data_types={"col": bolt.types.categorical(n_classes=40)},
        target="col",
        input_dim=200,
    )

    input_dim = udt_model._get_model().ops()[0].weights.shape[0]

    assert input_dim == 200


def test_udt_train_batch():
    import numpy as np

    model = bolt.UniversalDeepTransformer(
        data_types={
            "query": bolt.types.text(),
            "target": bolt.types.categorical(n_classes=3, type="int"),
        },
        target="target",
    )

    samples = [
        {"query": "this is zero", "target": "0"},
        {"query": "this is one", "target": "1"},
        {"query": "this is two", "target": "2"},
    ] * 1000

    for _ in range(3):
        model.train_batch(samples, learning_rate=0.1)

    scores = model.predict_batch(samples)

    predictions = np.argmax(scores, axis=0)

    assert (predictions == np.array([0, 1, 2])).all()


def test_model_dims_regular_udt():
    model = bolt.UniversalDeepTransformer(
        data_types={"col": bolt.types.categorical(n_classes=2)},
        target="col",
        input_dim=8,
        embedding_dimension=4,
    )

    assert model.model_dims() == [8, 4, 2]


def test_model_dims_mach():
    model = bolt.UniversalDeepTransformer(
        data_types={"col": bolt.types.categorical(n_classes=20, type="int")},
        target="col",
        input_dim=8,
        embedding_dimension=4,
        extreme_classification=True,
        extreme_num_hashes=1,
        extreme_output_dim=2,
    )

    assert model.model_dims() == [8, 4, 2]


def test_top_k_predictions():
    model = make_simple_trained_model(integer_label=False)
    predictions = sorted(model.predict(single_sample()), reverse=True)
    for topk in range(1, len(predictions)):
        topk_predictions = model.predict(single_sample(), top_k=topk)
        check_equal = all(
            [predictions[i] == topk_predictions[1][i] for i in range(topk)]
        )
        assert check_equal
