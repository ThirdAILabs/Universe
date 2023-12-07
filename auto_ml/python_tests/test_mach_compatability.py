import pytest
from thirdai import bolt
from download_dataset_fixtures import download_clinc_dataset


pytestmark = [pytest.mark.unit, pytest.mark.release]


def clinc_model():
    model = bolt.UniversalDeepTransformer(
        data_types={
            "category": bolt.types.categorical(),
            "text": bolt.types.text(),
        },
        target="category",
        n_target_classes=150,
        integer_target=True,
        options={
            "extreme_classification": True,
            "extreme_output_dim": 1000,
            "v1": True,
        },
    )

    return model


def test_mach_compatability(download_clinc_dataset):
    model = clinc_model()
    assert model.is_v1()

    train_filename, eval_filename, test_samples = download_clinc_dataset
    print(len(test_samples))
    test_samples = [x[0] for x in test_samples[:100]]

    model.train(train_filename, epochs=5, learning_rate=0.01)

    original_metrics = model.evaluate(eval_filename, metrics=["precision@1"])
    original_preds = model.predict_batch(test_samples, top_k=5)

    assert original_metrics["val_precision@1"][-1] >= 0.8

    save_path = "./mach_v1_model"

    model.save(save_path)

    model = bolt.UniversalDeepTransformer.load(save_path)
    assert not model.is_v1()

    new_metrics = model.evaluate(eval_filename, metrics=["precision@1"])
    new_preds = model.predict_batch(test_samples, top_k=5)

    assert new_metrics["val_precision@1"][-1] == original_metrics["val_precision@1"][-1]
    assert new_preds == original_preds

    model.train(train_filename, epochs=5, learning_rate=0.01)
    new_metrics = model.evaluate(eval_filename, metrics=["precision@1"])

    assert new_metrics["val_precision@1"][-1] > original_metrics["val_precision@1"][-1]
