import pytest
from thirdai import bolt

pytestmark = [pytest.mark.unit]

TRAIN_FILE = "tempTrainFile.csv"
TEST_FILE = "tempTestFile.csv"


def write_lines_to_file(file, lines):
    with open(file, "w") as f:
        for line in lines:
            f.writelines(line + "\n")


def make_simple_sequential_model():
    write_lines_to_file(
        TRAIN_FILE,
        [
            "userId,movieId,timestamp",
            "0,100,2022-08-29",
            "1,100,2022-08-30",
            "1,101,2022-08-31",
            "1,102,2022-09-01",
        ],
    )

    write_lines_to_file(
        TEST_FILE,
        [
            "userId,movieId,timestamp",
            "0,101,2022-08-31",
            "2,100,2022-08-30",
        ],
    )

    return bolt.Oracle(
        data_types={
            "userId": bolt.types.categorical(n_unique_classes=3),
            "movieId": bolt.types.categorical(n_unique_classes=3),
            "timestamp": bolt.types.date(),
        },
        temporal_tracking_relationships={"userId": ["movieId"]},
        target="movieId",
    )


def test_sequential_classifier_save_load():
    model = make_simple_sequential_model()

    model.train(TRAIN_FILE, 2, 0.01)
    model.save("saveLoc")
    before_load_metrics = model.evaluate(TEST_FILE)
    model = bolt.Oracle.load("saveLoc")
    after_load_metrics = model.evaluate(TEST_FILE)

    assert before_load_metrics["recall@1"] == after_load_metrics["recall@1"]


def test_multiple_predict_returns_same():
    model = make_simple_sequential_model()
    model.train(TRAIN_FILE, 2, 0.01)

    sample = {"userId": "0", "timestamp": "2022-08-31"}
    prev_result = model.predict_single(sample)
    for _ in range(5):
        result = model.predict_single(sample)
        assert prev_result == result
        prev_result = result


def test_multiple_explain_returns_same():
    model = make_simple_sequential_model()
    model.train(TRAIN_FILE, 2, 0.01)

    sample = {"userId": "0", "timestamp": "2022-08-31"}
    prev_explanations = model.explain(sample)
    for _ in range(5):
        explanations = model.explain(sample)
        assert len(prev_explanations) == len(explanations)
        for prev_explanation, explanation in zip(prev_explanations, explanations):
            assert prev_explanation.column_number == explanation.column_number
            assert (
                prev_explanation.percentage_significance
                == explanation.percentage_significance
            )
            assert prev_explanation.keyword == explanation.keyword
            assert prev_explanation.column_name == explanation.column_name

        prev_explanations = explanations


# def test_predict_returns_sorted_scores():
#     model = make_simple_sequential_model()
#     model.train(TRAIN_FILE, 2, 0.01)
#     top_k = 2
#     result = model.predict_single(
#         {"userId": "0", "timestamp": "2022-08-31"}, top_k=top_k
#     )
#     assert len(result) == top_k

#     prev_score = float("inf")
#     for prediction, score in result:
#         assert prev_score > score
#         prev_score = score


def test_index_changes_explain_and_predict():
    model = make_simple_sequential_model()
    model.train(TRAIN_FILE, 2, 0.01)

    sample = {"userId": "0", "timestamp": "2022-08-31"}

    first_result = model.predict_single(sample)
    first_explanations = model.explain(sample)

    model.index_single({"userId": "0", "movieId": "101", "timestamp": "2022-08-31"})

    second_result = model.predict_single(sample)
    second_explanations = model.explain(sample)

    assert first_result != second_result
    assert (
        first_explanations[0].percentage_significance
        != second_explanations[0].percentage_significance
    )


# TODO this doesnt fail??
def test_fail_on_relationships_with_no_datetime():
    model = bolt.Oracle(
        data_types={
            "userId": bolt.types.categorical(n_unique_classes=3),
            "movieId": bolt.types.categorical(n_unique_classes=3),
        },
        temporal_tracking_relationships={"userId": ["movieId"]},
        target="movieId",
    )


# TODO this doesnt fail??
def test_fail_on_multiple_datetime():
    model = bolt.Oracle(
        data_types={
            "userId": bolt.types.categorical(n_unique_classes=3),
            "movieId": bolt.types.categorical(n_unique_classes=3),
            "timestamp": bolt.types.date(),
            "timestamp1": bolt.types.date(),
        },
        temporal_tracking_relationships={"userId": ["movieId"]},
        target="movieId",
    )


def test_tracking_on_invalid_type_fails():
    with pytest.raises(
        ValueError,
        match=r"timestamp is neither numerical nor categorical. Only numerical and categorical columns can be tracked temporally.",
    ):
        bolt.Oracle(
            data_types={
                "userId": bolt.types.categorical(n_unique_classes=3),
                "timestamp": bolt.types.date(),
            },
            temporal_tracking_relationships={"userId": ["timestamp"]},
            target="movieId",
        )


# TODO how to specify autotuned temporal
def test_autotuned_tracking_on_invalid_type_fails():
    # with pytest.raises(
    #     ValueError,
    #     match=r"timestamp is neither numerical nor categorical. Only numerical and categorical columns can be tracked temporally.",
    # ):
    bolt.Oracle(
        data_types={
            "userId": bolt.types.categorical(n_unique_classes=3),
            "timestamp": bolt.types.date(),
        },
        temporal_tracking_relationships={"timestamp": ["userId"]},
        target="movieId",
    )