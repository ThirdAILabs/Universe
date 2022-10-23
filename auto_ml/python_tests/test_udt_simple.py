from random import sample
import pytest
from thirdai import bolt, deployment

pytestmark = [pytest.mark.unit]

TRAIN_FILE = "tempTrainFile.csv"
TEST_FILE = "tempTestFile.csv"


def write_lines_to_file(file, lines):
    with open(file, "w") as f:
        for line in lines:
            f.writelines(line + "\n")


def make_simple_trained_model(embedding_dim=None):
    write_lines_to_file(
        TRAIN_FILE,
        [
            "userId,movieId,timestamp,hoursWatched",
            "0,0,2022-08-29,2",
            "1,0,2022-08-30,3",
            "1,1,2022-08-31,1",
            "1,2,2022-09-01,3",
        ],
    )

    write_lines_to_file(
        TEST_FILE,
        [
            "userId,movieId,timestamp,hoursWatched",
            "0,1,2022-08-31,5",
            "2,0,2022-08-30,0.5",
        ],
    )

    model = deployment.UniversalDeepTransformer(
        data_types={
            "userId": bolt.types.categorical(n_unique_classes=3),
            "movieId": bolt.types.categorical(n_unique_classes=3),
            "timestamp": bolt.types.date(),
            "hoursWatched": bolt.types.numerical(),
        },
        temporal_tracking_relationships={"userId": ["movieId", "hoursWatched"]},
        target="movieId",
        options={"embedding_dimension": str(embedding_dim)} if embedding_dim else {},
    )

    train_config = bolt.graph.TrainConfig.make(epochs=2, learning_rate=0.01)
    model.train(TRAIN_FILE, train_config, batch_size=2048)

    return model


def single_sample():
    return {"userId": "0", "timestamp": "2022-08-31"}


def batch_sample():
    return [single_sample(), single_sample(), single_sample()]


def single_update():
    return {
        "userId": "0",
        "movieId": "1",
        "timestamp": "2022-08-31",
        "hoursWatched": "1",
    }


def batch_update():
    return [single_update(), single_update(), single_update()]


def assert_explanations_equal(explanations_1, explanations_2):
    assert len(explanations_1) == len(explanations_2)
    for exp_1, exp_2 in zip(explanations_1, explanations_2):
        assert exp_1.column_number == exp_2.column_number
        assert exp_1.column_name == exp_2.column_name
        assert exp_1.percentage_significance == exp_2.percentage_significance
        assert exp_1.keyword == exp_2.keyword


def test_save_load():
    save_file = "savefile.bolt"
    model = make_simple_trained_model()
    model.save(save_file)
    saved_model = deployment.UniversalDeepTransformer.load(save_file)

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
    assert_explanations_equal(explain_res, saved_explain_res)


def test_multiple_predict_returns_same_results():
    model = make_simple_trained_model()
    first = model.predict(single_sample())
    second = model.predict(single_sample())
    assert (first == second).all()

    first_batch = model.predict_batch(batch_sample())
    second_batch = model.predict_batch(batch_sample())
    assert (first_batch == second_batch).all()


def test_index_changes_predict_result():
    model = make_simple_trained_model()
    first = model.predict(single_sample())
    model.index(single_update())
    second = model.predict(single_sample())
    assert (first != second).any()

    model.index_batch(batch_update())
    third = model.predict(single_sample())
    assert (second != third).any()


def test_embedding_representation_returns_correct_dimension():
    for embedding_dim in [256, 512, 1024]:
        model = make_simple_trained_model(embedding_dim)
        embedding = model.embedding_representation(single_sample())
        assert embedding.shape == (embedding_dim,)
        assert (embedding != 0).any()


def test_explanations_total_percentage():
    model = make_simple_trained_model()
    explanations = model.explain(single_sample())
    total_percentage = 0
    for explanation in explanations:
        total_percentage += abs(explanation.percentage_significance)

    assert total_percentage > 99.99


def test_reset_clears_history():
    model = make_simple_trained_model()
    model.reset_temporal_trackers()
    first = model.predict(single_sample())

    model.index(single_update())
    after_index = model.predict(single_sample())
    assert (first != after_index).any()

    model.reset_temporal_trackers()
    after_reset = model.predict(single_sample())
    assert (first == after_reset).all()
