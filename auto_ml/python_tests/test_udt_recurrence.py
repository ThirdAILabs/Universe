import os
import platform

import pytest
from thirdai import bolt

pytestmark = [pytest.mark.unit]

TRAIN_FILE = "./udt_recursive_data.csv"


def recursive_model(
    inputs=["1 2 3 4", "3 4 5"],
    outputs=["1\t2\t3\t4", "3\t4\t5"],
    output_delimiter="\t",
    n_target_classes=5,
):
    data = ["input,output\n", *[f"{inp},{out}\n" for inp, out in zip(inputs, outputs)]]

    with open(TRAIN_FILE, "w") as file:
        file.writelines(data)

    model = bolt.UniversalDeepTransformer(
        data_types={
            "input": bolt.types.sequence(),  # Delimiter defaults to " "
            "output": bolt.types.sequence(max_length=4, delimiter=output_delimiter),
        },
        target="output",
        n_target_classes=n_target_classes,
    )

    model.train(TRAIN_FILE, learning_rate=0.01, epochs=10, verbose=False)

    return model


def test_udt_recurrence_save_load():
    model = recursive_model()
    metrics_before_save = model.evaluate(TRAIN_FILE, verbose=False)
    prediction_before_save = model.predict({"input": "1 2 3 4"})
    model.save("save.bolt")
    saved_model = bolt.UniversalDeepTransformer.load("save.bolt")
    metrics_after_save = saved_model.evaluate(TRAIN_FILE, verbose=False)
    prediction_after_save = saved_model.predict({"input": "1 2 3 4"})

    assert metrics_before_save == metrics_after_save
    assert prediction_before_save == prediction_after_save


def test_udt_recurrence_predict():
    model = recursive_model()
    prediction = model.predict({"input": "1 2 3 4"})
    assert prediction == "1\t2\t3\t4"
    os.remove(TRAIN_FILE)


def test_udt_recurrence_predict_batch():
    model = recursive_model()
    predictions = model.predict_batch([{"input": "1 2 3 4"}, {"input": "3 4 5"}])
    assert predictions[0] == "1\t2\t3\t4"
    assert predictions[1] == "3\t4\t5"
    os.remove(TRAIN_FILE)


def test_udt_recurrence_return_metrics():
    model = recursive_model()
    metrics = model.train(
        TRAIN_FILE, learning_rate=0.01, epochs=10, metrics=["categorical_accuracy"]
    )

    assert metrics["train_categorical_accuracy"][-1] > 0

    metrics = model.evaluate(TRAIN_FILE, metrics=["categorical_accuracy"])
    assert metrics["val_categorical_accuracy"][-1] > 0

    os.remove(TRAIN_FILE)


def test_udt_recurrence_long_output_does_not_break():
    model = recursive_model(
        inputs=["1 2 3 4", "3 4 5"],
        outputs=["1 2 3 4 5 6 7", "3 4 5 6 7 8 9"],
        output_delimiter=" ",
        n_target_classes=9,
    )
    predictions = model.predict_batch([{"input": "1 2 3 4"}, {"input": "3 4 5"}])
    assert predictions == ["1 2 3 4", "3 4 5 6"]

    os.remove(TRAIN_FILE)


def test_udt_recurrence_long_output_ignores_remaining():
    model = recursive_model(
        inputs=["1 2 3 4", "3 4 5"],
        outputs=["1 2 3 4 5 6 7", "3 4 5 6 7 8 9"],
        output_delimiter=" ",
        # If not ignored, it will throw since there are 9 unique output classes
        n_target_classes=6,
    )
    predictions = model.predict_batch([{"input": "1 2 3 4"}, {"input": "3 4 5"}])
    assert predictions == ["1 2 3 4", "3 4 5 6"]

    os.remove(TRAIN_FILE)


def test_udt_recurrence_short_output_does_not_break():
    model = recursive_model(
        inputs=["1 2 3 4", "3 4 5"],
        outputs=["1 2", "3 4"],
        output_delimiter=" ",
    )
    predictions = model.predict_batch([{"input": "1 2 3 4"}, {"input": "3 4 5"}])
    assert predictions == ["1 2", "3 4"]

    os.remove(TRAIN_FILE)


def test_udt_recurrence_target_no_max_length_throws():
    with pytest.raises(
        ValueError, match="Must provide max_length for target sequence."
    ):
        model = bolt.UniversalDeepTransformer(
            data_types={
                "input": bolt.types.sequence(),
                "output": bolt.types.sequence(),
            },
            target="output",
            n_target_classes=4,
        )


def test_udt_recurrence_zero_max_length_throws():
    with pytest.raises(
        ValueError,
        match=f"Sequence max_length cannot be 0.",
    ):
        model = bolt.UniversalDeepTransformer(
            data_types={
                "input": bolt.types.sequence(),
                "output": bolt.types.sequence(max_length=0),
            },
            target="output",
            n_target_classes=4,
        )
