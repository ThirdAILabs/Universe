import platform

import pytest
from thirdai import bolt

pytestmark = [pytest.mark.unit, pytest.mark.release]

TRAIN_FILE = "tempTrainFile.csv"
TEST_FILE = "tempTestFile.csv"
METADATA_FILE = "tempMetaFile.csv"


def write_lines_to_file(file, lines):
    with open(file, "w") as f:
        for line in lines:
            f.writelines(line + "\n")


def make_simple_trained_model(embedding_dim=None, integer_label=False):
    write_lines_to_file(
        TRAIN_FILE,
        [
            "userId,movieId,timestamp,hoursWatched,genres,meta,description",
            "0,0,2022-08-29,2,fiction-comedy-drama,0-1,a movie",
            "1,0,2022-08-30,2,fiction-romance,1,a movie",
            "1,1,2022-08-31,1,romance-comedy,0,a movie",
            # if integer_label = false, we build a model that accepts
            # arbitrary string labels; the model does not expect integer
            # labels in the range [0, n_labels - 1]. We test this by
            # checking that the model does not throw an error when given
            # a label outside of this range. Since n_labels = 3, we set
            # movieId = 4 in the last sample and expect that the model
            # trains just fine.
            (
                "1,2,2022-09-01,3,fiction-comedy,1-2,a movie"
                if integer_label
                else "1,4,2022-09-01,3,fiction-comedy,1-4,a movie"
            ),
        ],
    )

    write_lines_to_file(
        TEST_FILE,
        [
            "userId,movieId,timestamp,hoursWatched,genres,meta,description",
            "0,1,2022-08-31,5,fiction-drama,0,a movie",
            # See above comment about the last line of the mock train file.
            (
                "1,0,2022-09-01,0.5,fiction-comedy,2-0,a movie"
                if integer_label
                else "4,0,2022-09-01,0.5,fiction-comedy,4-0,a movie"
            ),
        ],
    )

    keys = [0, 1, 2] if integer_label else [0, 1, 4]
    metadata_lines = [str(key) + "," + str(val) for key, val in zip(keys, [1, 2, 3])]
    write_lines_to_file(METADATA_FILE, ["id,feature"] + metadata_lines)

    metadata = bolt.types.metadata(
        filename=METADATA_FILE,
        key_column_name="id",
        data_types={"feature": bolt.types.categorical()},
    )

    model = bolt.UniversalDeepTransformer(
        data_types={
            "userId": bolt.types.categorical(metadata=metadata),
            "movieId": bolt.types.categorical(
                metadata=metadata,
            ),
            "timestamp": bolt.types.date(),
            "hoursWatched": bolt.types.numerical(range=(0, 5)),
            "genres": bolt.types.categorical(delimiter="-"),
            "meta": bolt.types.categorical(metadata=metadata, delimiter="-"),
            "description": bolt.types.text(),
        },
        temporal_tracking_relationships={"userId": ["movieId", "hoursWatched"]},
        target="movieId",
        n_target_classes=3,
        integer_target=integer_label,
        options={"embedding_dimension": str(embedding_dim)} if embedding_dim else {},
    )

    model.train(TRAIN_FILE, epochs=2, learning_rate=0.01, batch_size=2048)

    return model


def single_sample():
    return {
        "userId": "0",
        "timestamp": "2022-08-31",
        "genres": "fiction-drama",
        "meta": "0",
    }


def batch_sample():
    return [single_sample(), single_sample(), single_sample()]


def single_update():
    return {
        "userId": "0",
        "movieId": "1",
        "timestamp": "2022-08-31",
        "hoursWatched": "1",
        "genres": "fiction-drama",
        "meta": "0",
    }


def batch_update():
    return [single_update(), single_update(), single_update()]


def compare_explanations(explanations_1, explanations_2, assert_mode):
    all_equal = len(explanations_1) == len(explanations_2)
    for exp_1, exp_2 in zip(explanations_1, explanations_2):
        all_equal = all_equal and (
            (exp_1.column_name == exp_2.column_name)
            and (exp_1.percentage_significance == exp_2.percentage_significance)
            and (exp_1.keyword == exp_2.keyword)
        )

    # If we want to assert equality, we want everything to be equal
    # Otherwise, we want something to be inequal.
    assert_equal = assert_mode == "equal"
    assert assert_equal == all_equal


def test_save_load():
    save_file = "savefile.bolt"
    model = make_simple_trained_model(integer_label=False)
    model.save(save_file)
    saved_model = bolt.UniversalDeepTransformer.load(filename=save_file)

    eval_res = model.evaluate(TEST_FILE)
    saved_eval_res = saved_model.evaluate(TEST_FILE)
    assert (eval_res == saved_eval_res).all()

    model.index(single_update())
    saved_model.index(single_update())

    predict_res = model.predict(single_sample())
    saved_predict_res = saved_model.predict(single_sample())
    assert (predict_res == saved_predict_res).all()

    predict_batch_res = model.predict_batch(batch_sample())
    saved_predict_batch_res = saved_model.predict_batch(batch_sample())
    assert (predict_batch_res == saved_predict_batch_res).all()

    explain_res = model.explain(single_sample())
    saved_explain_res = saved_model.explain(single_sample())
    compare_explanations(explain_res, saved_explain_res, assert_mode="equal")


def test_multiple_predict_returns_same_results():
    model = make_simple_trained_model(integer_label=False)
    first = model.predict(single_sample())
    second = model.predict(single_sample())
    assert (first == second).all()

    first_batch = model.predict_batch(batch_sample())
    second_batch = model.predict_batch(batch_sample())
    assert (first_batch == second_batch).all()


def test_index_changes_predict_result():
    model = make_simple_trained_model(integer_label=False)
    first = model.predict(single_sample())
    model.index(single_update())
    second = model.predict(single_sample())
    assert (first != second).any()

    model.index_batch(batch_update())
    third = model.predict(single_sample())
    assert (second != third).any()


def test_embedding_representation_returns_correct_dimension():
    for embedding_dim in [128, 256]:
        model = make_simple_trained_model(embedding_dim=embedding_dim)
        embedding = model.embedding_representation(single_sample())
        assert embedding.shape == (embedding_dim,)
        assert (embedding != 0).any()


def test_explanations_total_percentage():
    model = make_simple_trained_model(integer_label=False)
    explanations = model.explain(single_sample())
    total_percentage = 0
    for explanation in explanations:
        total_percentage += abs(explanation.percentage_significance)

    assert total_percentage > 99.99 and total_percentage < 100.01


def test_different_explanation_target_returns_different_results():
    model = make_simple_trained_model(integer_label=False)

    explain_target_1 = model.explain(single_sample(), target_class="1")
    explain_target_2 = model.explain(single_sample(), target_class="4")
    compare_explanations(explain_target_1, explain_target_2, assert_mode="not_equal")


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="Throwing an exception leads to an access violation on windows.",
)
def test_explanations_target_label_format():
    model = make_simple_trained_model(integer_label=False)
    # Call this method to make sure it does not throw an error
    model.explain(single_sample(), target_class="1")
    with pytest.raises(ValueError, match=r"Received an integer but*"):
        model.explain(single_sample(), target_class=1)

    model = make_simple_trained_model(integer_label=True)
    # Call this method to make sure it does not throw an error
    model.explain(single_sample(), target_class=1)
    with pytest.raises(ValueError, match=r"Received a string but*"):
        model.explain(single_sample(), target_class="1")


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
            "movieId": bolt.types.categorical(),
            "hoursWatched": bolt.types.numerical(range=(0, 5)),
            "genres": bolt.types.categorical(delimiter="-"),
        },
        target="movieId",
        n_target_classes=3,
    )

    model.train(TRAIN_FILE, epochs=2, learning_rate=0.01, batch_size=2048)
    model.evaluate(TEST_FILE)

    # No assertion as we just want to know that there is no error.
