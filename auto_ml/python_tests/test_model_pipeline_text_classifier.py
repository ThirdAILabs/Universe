import pytest
from download_datasets import download_clinc_dataset
from model_test_utils import (
    check_saved_and_retrained_accuarcy,
    compute_evaluate_accuracy,
    compute_predict_accuracy,
    compute_predict_batch_accuracy,
)
from thirdai import bolt, deployment

pytestmark = [pytest.mark.integration, pytest.mark.release]

ACCURACY_THRESHOLD = 0.8


@pytest.fixture(scope="module")
def download_clinc_dataset_model_pipeline(download_clinc_dataset):
    train, test, inference = download_clinc_dataset
    inference = [(x["text"], y) for x, y in inference]
    return train, test, inference


@pytest.fixture(scope="module")
def train_model_pipeline_text_classifier(download_clinc_dataset_model_pipeline):
    model_config = deployment.ModelConfig(
        input_names=["input"],
        nodes=[
            deployment.FullyConnectedNodeConfig(
                name="hidden",
                dim=deployment.OptionMappedParameter(
                    option_name="size", values={"small": 100, "large": 512}
                ),
                activation=deployment.ConstantParameter("relu"),
                predecessor="input",
            ),
            deployment.FullyConnectedNodeConfig(
                name="output",
                dim=deployment.UserSpecifiedParameter("output_dim", type=int),
                sparsity=deployment.ConstantParameter(1.0),
                activation=deployment.ConstantParameter("softmax"),
                predecessor="hidden",
            ),
        ],
        loss=bolt.nn.losses.CategoricalCrossEntropy(),
    )

    dataset_config = deployment.SingleBlockDatasetFactory(
        data_block=deployment.TextBlockConfig(use_pairgrams=False),
        label_block=deployment.NumericalCategoricalBlockConfig(
            n_classes=deployment.UserSpecifiedParameter("output_dim", type=int),
            delimiter=deployment.ConstantParameter(","),
        ),
        shuffle=deployment.ConstantParameter(True),
        delimiter=deployment.UserSpecifiedParameter("delimiter", type=str),
        has_header=True,
    )

    train_eval_params = deployment.TrainEvalParameters(
        rebuild_hash_tables_interval=None,
        reconstruct_hash_functions_interval=None,
        default_batch_size=2048,
        freeze_hash_tables=False,
    )

    config = deployment.DeploymentConfig(
        dataset_config=dataset_config,
        model_config=model_config,
        train_eval_parameters=train_eval_params,
    )

    CONFIG_FILE = "./serialized_model_config"
    config.save(CONFIG_FILE)

    model = bolt.models.Pipeline(
        config_path=CONFIG_FILE,
        parameters={"size": "large", "output_dim": 150, "delimiter": ","},
    )

    train_filename, _, _ = download_clinc_dataset_model_pipeline

    model.train(
        filename=train_filename,
        learning_rate=0.01,
        epochs=5,
        max_in_memory_batches=12,
    )

    return model


def test_model_pipeline_text_classifier_accuracy(
    train_model_pipeline_text_classifier, download_clinc_dataset_model_pipeline
):
    model = train_model_pipeline_text_classifier
    _, test_filename, inference_samples = download_clinc_dataset_model_pipeline

    acc = compute_evaluate_accuracy(
        model, test_filename, inference_samples, use_class_name=False
    )
    assert acc >= ACCURACY_THRESHOLD


def test_model_pipeline_text_classifier_predict_single(
    train_model_pipeline_text_classifier, download_clinc_dataset_model_pipeline
):
    model = train_model_pipeline_text_classifier
    _, _, inference_samples = download_clinc_dataset_model_pipeline

    acc = compute_predict_accuracy(model, inference_samples, use_class_name=False)
    assert acc >= ACCURACY_THRESHOLD


def test_model_pipeline_text_classifier_predict_batch(
    train_model_pipeline_text_classifier, download_clinc_dataset_model_pipeline
):
    model = train_model_pipeline_text_classifier
    _, _, inference_samples = download_clinc_dataset_model_pipeline

    acc = compute_predict_batch_accuracy(model, inference_samples, use_class_name=False)
    assert acc >= ACCURACY_THRESHOLD


def test_model_pipeline_text_classification_save_load(
    train_model_pipeline_text_classifier, download_clinc_dataset_model_pipeline
):
    model = train_model_pipeline_text_classifier
    (
        train_filename,
        test_filename,
        inference_samples,
    ) = download_clinc_dataset_model_pipeline

    check_saved_and_retrained_accuarcy(
        model,
        train_filename,
        test_filename,
        inference_samples,
        use_class_name=False,
        accuracy=ACCURACY_THRESHOLD,
        model_type="Pipeline",
    )


# Because validatation doesn't return anything there isn't anything specific to test
# here, this is just a sanity check that using validation produces no errors.
def test_model_pipeline_text_classification_train_with_validation(
    train_model_pipeline_text_classifier, download_clinc_dataset_model_pipeline
):
    model = train_model_pipeline_text_classifier
    train_filename, test_filename, _ = download_clinc_dataset_model_pipeline

    validation = bolt.Validation(
        filename=test_filename,
        interval=4,
        metrics=["categorical_accuracy"]
    )

    model.train(
        filename=train_filename,
        epochs=1,
        learning_rate=0.001,
        validation=validation,
    )
