import pytest
from thirdai import deployment_config as dc
from thirdai import bolt


def test_text_classifer():
    model_config = dc.ModelConfig(
        input_names=["input"],
        nodes=[
            dc.FullyConnectedNodeConfig(
                name="hidden",
                dim=dc.OptionParameter({"small": 100, "large": 200}),
                activation=dc.ConstantParameter("relu"),
                predecessor="input"
            ),
            dc.FullyConnectedNodeConfig(
                name="output",
                dim=dc.UserSpecifiedParameter("output_dim"),
                activation=dc.ConstantParameter("softmax"),
                predecessor="hidden"
            )
        ],
        loss=dc.ConstantParameter(bolt.CategoricalCrossEntropyLoss())
    )

    dataset_config = dc.BasicClassificationDatasetConfig(
        data_block=dc.TextBlockConfig(
            use_pairgrams=True,
            range=100_000
        ),
        label_block=dc.NumericalCategoricalBlockConfig(
            n_classes=dc.UserSpecifiedParameter("output_dim")
        ),
        delimiter=dc.UserSpecifiedParameter("delimiter")
    )

    train_eval_params = dc.TrainEvalParameters(
        rebuild_hash_tables_interval=None,
        reconstruct_hash_functions_interval=None,
        default_batch_size=256,
        use_sparse_inference=True,
        evaluation_metrics=["categorical_accuracy"]
    )

    config = dc.DeploymentConfig(
        dataset_config=dataset_config, model_config=model_config, train_eval_parameters=train_eval_params
    )

    model = dc.ModelPipeline(
      config=config,
      size="large",
      parameters={
        "output_dim": 151, "delimiter": ','
      }
    )