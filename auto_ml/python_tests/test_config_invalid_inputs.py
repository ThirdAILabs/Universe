import pytest
from thirdai import bolt, deployment

pytestmark = [pytest.mark.unit]


def get_config(
    add_extra_input=False,
    missing_predecessor=False,
    duplicate_node_name=False,
):
    if add_extra_input:
        inputs = ["input", "input_2"]
    else:
        inputs = ["input"]

    if missing_predecessor:
        predecessor = "missing_node"
    else:
        predecessor = "input"

    nodes = [
        deployment.FullyConnectedNodeConfig(
            name="output",
            dim=deployment.UserSpecifiedParameter("output_dim", type=int),
            sparsity=deployment.OptionMappedParameter(
                option_name="sparsity_level", values={"sparse": 0.5, "dense": 1.0}
            ),
            activation=deployment.ConstantParameter("softmax"),
            predecessor=predecessor,
        ),
    ]

    if duplicate_node_name:
        nodes.append(nodes[0])

    model_config = deployment.ModelConfig(
        input_names=inputs,
        nodes=nodes,
        loss=bolt.CategoricalCrossEntropyLoss(),
    )

    dataset_config = deployment.SingleBlockDatasetFactory(
        data_block=deployment.TextBlockConfig(use_pairgrams=True),
        label_block=deployment.NumericalCategoricalBlockConfig(
            n_classes=deployment.UserSpecifiedParameter("output_dim", type=int),
            delimiter=deployment.ConstantParameter(","),
        ),
        shuffle=deployment.ConstantParameter(False),
        delimiter=deployment.UserSpecifiedParameter("delimiter", type=str),
    )

    train_eval_params = deployment.TrainEvalParameters(
        rebuild_hash_tables_interval=None,
        reconstruct_hash_functions_interval=None,
        default_batch_size=100,
        freeze_hash_tables=True,
    )

    config = deployment.DeploymentConfig(
        dataset_config=dataset_config,
        model_config=model_config,
        train_eval_parameters=train_eval_params,
    )

    return config


def test_missing_parameter_throws():
    with pytest.raises(
        ValueError,
        match=r"UserSpecifiedParameter 'output_dim' not specified by user but is required to construct ModelPipeline.",
    ):
        deployment.ModelPipeline(
            deployment_config=get_config(),
            parameters={"sparsity_level": "sparse", "delimiter": ","},
        )


def test_wrong_type_parameter_throws():
    with pytest.raises(
        ValueError, match=r"Expected parameter 'output_dim'to be of type int."
    ):
        deployment.ModelPipeline(
            deployment_config=get_config(),
            parameters={
                "output_dim": 4.2,
                "sparsity_level": "sparse",
                "delimiter": ",",
            },
        )


def test_missing_option_parameter_throws():
    with pytest.raises(
        ValueError,
        match=r"UserSpecifiedParameter 'sparsity_level' not specified by user but is required to construct ModelPipeline.",
    ):
        deployment.ModelPipeline(
            deployment_config=get_config(),
            parameters={"output_dim": 100, "delimiter": ","},
        )


def test_invalid_option_parameter_option():
    with pytest.raises(
        ValueError,
        match=r"Invalid option 'sort-of-sparse' for 'sparsity_level'. Supported options are: \[ 'dense' 'sparse' \].",
    ):
        deployment.ModelPipeline(
            deployment_config=get_config(),
            parameters={
                "output_dim": 100,
                "sparsity_level": "sort-of-sparse",
                "delimiter": ",",
            },
        )


def test_invalid_parameter_type_throws():
    with pytest.raises(
        ValueError,
        match=r"Invalid type '<class 'list'>'. Values of parameters dictionary must be bool, int, float, or str.",
    ):
        deployment.ModelPipeline(
            deployment_config=get_config(),
            parameters={"output_dim": [], "sparsity_level": "sparse", "delimiter": ","},
        )


def test_input_mismatch_throws():
    with pytest.raises(
        ValueError,
        match=r"Number of inputs in model config does not match number of inputs returned from data loader.",
    ):
        deployment.ModelPipeline(
            deployment_config=get_config(add_extra_input=True),
            parameters={
                "output_dim": 100,
                "sparsity_level": "sparse",
                "delimiter": ",",
            },
        )


def test_duplicate_node_name_throws():
    with pytest.raises(
        ValueError,
        match=r"Cannot have multiple nodes with the name 'output' in the model config.",
    ):
        deployment.ModelPipeline(
            deployment_config=get_config(duplicate_node_name=True),
            parameters={
                "output_dim": 100,
                "sparsity_level": "sparse",
                "delimiter": ",",
            },
        )


def test_missing_predecessor_throws():
    with pytest.raises(
        ValueError,
        match=r"Cannot find node with name 'missing_node' in already discovered nodes.",
    ):
        deployment.ModelPipeline(
            deployment_config=get_config(missing_predecessor=True),
            parameters={
                "output_dim": 100,
                "sparsity_level": "sparse",
                "delimiter": ",",
            },
        )


def test_invalid_delimiter_throws():
    with pytest.raises(
        ValueError,
        match=r"Expected delimiter to be a single character but recieved: ',,'.",
    ):
        deployment.ModelPipeline(
            deployment_config=get_config(),
            parameters={
                "output_dim": 100,
                "sparsity_level": "sort-of-sparse",
                "delimiter": ",,",
            },
        )
