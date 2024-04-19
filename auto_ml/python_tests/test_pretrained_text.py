import pytest
from download_dataset_fixtures import download_clinc_dataset
from model_test_utils import (
    check_saved_and_retrained_accuarcy,
    compute_evaluate_accuracy,
    compute_predict_accuracy,
    compute_predict_batch_accuracy,
)
from thirdai import bolt, dataset

ACCURACY_THRESHOLD = 0.8

pytestmark = [pytest.mark.unit]


def build_dummy_pretrained_model():
    input_ = bolt.nn.Input(dim=20)
    out = bolt.nn.FullyConnected(dim=10, input_dim=20, activation="softmax")(input_)
    loss = bolt.nn.losses.CategoricalCrossEntropy(out, bolt.nn.Input(10))
    model = bolt.nn.Model(inputs=[input_], outputs=[out], losses=[loss])

    index = dataset.MachIndex(output_range=10, num_hashes=1, num_elements=20, seed=14)

    tokenizer = dataset.NaiveSplitTokenizer(" ")

    return bolt.SpladeMach("text", models=[model], indexes=[index], tokenizer=tokenizer)


def clinc_model():

    model = bolt.UniversalDeepTransformer(
        data_types={
            "category": bolt.types.categorical(),
            "text": bolt.types.text(),
        },
        pretrained_model=build_dummy_pretrained_model(),
        n_target_classes=150,
        integer_target=True,
    )

    return model


@pytest.fixture(scope="module")
def train_udt_pretrained_text(download_clinc_dataset):
    model = clinc_model()

    train_filename, _, _ = download_clinc_dataset

    model.cold_start(
        train_filename,
        strong_column_names=["text"],
        weak_column_names=[],
        epochs=1,
        learning_rate=0.01,
    )
    model.train(train_filename, epochs=4, learning_rate=0.01)

    return model


def test_udt_pretrained_text_accuarcy(
    train_udt_pretrained_text, download_clinc_dataset
):
    model = train_udt_pretrained_text
    _, test_filename, _ = download_clinc_dataset

    assert compute_evaluate_accuracy(model, test_filename) >= ACCURACY_THRESHOLD


def test_udt_pretrained_text_save_load(
    train_udt_pretrained_text, download_clinc_dataset
):
    model = train_udt_pretrained_text
    train_filename, test_filename, inference_samples = download_clinc_dataset

    check_saved_and_retrained_accuarcy(
        model, train_filename, test_filename, accuracy=ACCURACY_THRESHOLD
    )


def test_udt_pretrained_text_predict_single(
    train_udt_pretrained_text, download_clinc_dataset
):
    model = train_udt_pretrained_text
    _, _, inference_samples = download_clinc_dataset

    acc = compute_predict_accuracy(model, inference_samples, use_class_name=False)
    assert acc >= ACCURACY_THRESHOLD


def test_udt_pretrained_text_predict_batch(
    train_udt_pretrained_text, download_clinc_dataset
):
    model = train_udt_pretrained_text
    _, _, inference_samples = download_clinc_dataset

    acc = compute_predict_batch_accuracy(model, inference_samples, use_class_name=False)
    assert acc >= ACCURACY_THRESHOLD


def test_udt_pretrained_text_invalid_data_types():
    error = (
        "Expected only a text input and categorial output to use pretrained classifier."
    )

    bad_dtypes = [
        {"a": bolt.types.numerical((1, 2)), "b": bolt.types.text()},
        {"a": bolt.types.categorical(), "b": bolt.types.numerical((1, 2))},
        {
            "a": bolt.types.categorical(),
            "b": bolt.types.numerical((1, 2)),
            "c": bolt.types.text(),
        },
    ]

    for dtypes in bad_dtypes:
        with pytest.raises(ValueError, match=error):
            bolt.UniversalDeepTransformer(
                data_types=dtypes,
                pretrained_model=None,
                n_target_classes=150,
                integer_target=True,
            )
