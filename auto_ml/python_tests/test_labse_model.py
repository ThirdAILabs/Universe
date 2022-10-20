import pytest
import numpy as np
from thirdai import bolt, deployment


pytestmark = [pytest.mark.integration, pytest.mark.release]

TRAIN_FILE = "./data_labse_bal.csv"
TEST_FILE = "./test_labse.csv"
CONFIG_FILE = ""
INPUT_DIM = 100000


@pytest.fixture("module")
def trained_labse_model():
    model_config = deployment.ModelConfig(
        input_names=["input1", "input2"],
        nodes=[
            deployment.FullyConnectedNodeConfig(
                name="hidden",
                dim=deployment.OptionMappedParameter(
                    option_name="size", values={"small": 100, "large": 200}
                ),
                sparsity=deployment.ConstantParameter(0.1),
                activation=deployment.ConstantParameter("relu"),
                predecessor="input1"
            ),
            deployment.FullyConnectedNodeConfig(
                name="hidden",
                dim=deployment.OptionMappedParameter(
                    option_name="size", values={"small": 100, "large": 200}
                ),
                sparsity=deployment.ConstantParameter(0.1),
                activation=deployment.ConstantParameter("relu"),
                predecessor="hidden_enu"
            ),
            deployment.FullyConnectedNodeConfig(
                name="hidden",
                dim=deployment.OptionMappedParameter(
                    option_name="size", values={"small": 100, "large": 200}
                ),
                sparsity=deployment.ConstantParameter(0.1),
                activation=deployment.ConstantParameter("relu"),
                predecessor="input2"
            ),
            deployment.FullyConnectedNodeConfig(
                name="hidden",
                dim=deployment.OptionMappedParameter(
                    option_name="size", values={"small": 100, "large": 200}
                ),
                sparsity=deployment.ConstantParameter(0.1),
                activation=deployment.ConstantParameter("relu"),
                predecessor="input1"
            ),
        ],
        loss=bolt.MarginBCE()
    )

    dataset_config = deployment.DoubleBlockDatasetFactory(
        data_block=deployment.TextBlockConfig(use_pairgrams=True),
        label_block=deployment.NumericalCategoricalBlockConfig(
            n_classes=deployment.UserSpecifiedParameter("output_dim", type=int),
            delimeter=deployment.ConstantParameter("\t"),
        ),
        shuffle=deployment.ConstantParameter(False),
        delimeter=deployment.UserSpecifiedParameter("delimeter", type=str),
    )

    train_eval_params = deployment.TrainEvalParameters(
        rebuild_hash_tables_interval=None,
        reconstruct_hash_functions_interval=None,
        default_batch_size=256,
        freeze_hash_tables=True,
    )

    config = deployment.DeploymentConfig(
        dataset_config=dataset_config,
        model_config=model_config,
        train_eval_parameters=train_eval_params
    )

    config.save(CONFIG_FILE)

    model = deployment.ModelPipeline(
        config_path=CONFIG_FILE,
        parameters={"size": "large", "output_dim": 2, "delimeter": ","}
    )

    train_config = (
        bolt.graph.TrainConfig
            .make(learning_rate=0.0001, epochs=1)
            .with_metrics(["mean_squared_error"])
    )

    model.train(
        filename=TRAIN_FILE,
        train_config=train_config,
        batch_size=256,
        max_in_memory_batches=12
    )

    return model


def test_labse_model(trained_labse_model):
    predict_config = (
        bolt.graph.PredictConfig
            .make()
            .with_metrics(["mean_squared_error"])
            .return_activations()
    )
    logits = trained_labse_model.evaluate(
        filename=TEST_FILE,
        predict_config=predict_config
    )
    predictions = np.argmax(logits, axis=1)

    return predictions 






