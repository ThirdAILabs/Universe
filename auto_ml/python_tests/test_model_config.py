import json
import os
import re
import textwrap

import pytest
from thirdai import bolt


def get_config(have_user_specified_parameters: bool = False):
    if have_user_specified_parameters:
        layer_2_sparsity = {
            "param_name": "use_sparsity",
            "param_options": {"sparse": 0.25, "dense": 1.0},
        }
        layer_3_activation = {"param_name": "act"}
    else:
        layer_2_sparsity = 0.5
        layer_3_activation = "relu"
    config = {
        "inputs": ["input"],
        "nodes": [
            {
                "name": "fc_1",
                "type": "fully_connected",
                "dim": 10,
                "sparsity": 1.0,
                "activation": "tanh",
                "predecessor": "input",
            },
            {
                "name": "fc_2",
                "type": "fully_connected",
                "dim": 20,
                "sparsity": layer_2_sparsity,
                "activation": "relu",
                "sampling_config": "random",
                "predecessor": "fc_1",
            },
            {
                "name": "fc_3",
                "type": "fully_connected",
                "dim": 30,
                "sparsity": 0.3,
                "activation": layer_3_activation,
                "predecessor": "fc_2",
            },
            {
                "name": "fc_4",
                "type": "fully_connected",
                "dim": {"param_name": "output_dim"},
                "sparsity": 0.1,
                "activation": "softmax",
                "sampling_config": {
                    "num_tables": 4,
                    "hashes_per_table": 2,
                    "range_pow": 6,
                    "binsize": 8,
                    "reservoir_size": 10,
                    "permutations": 8,
                },
                "predecessor": "fc_3",
            },
        ],
        "output": "fc_4",
        "loss": "CategoricalCrossEntropyLoss",
    }

    return config


def compare_summaries(model, expected_summary):
    summary = model.summary(print=False)
    summary = textwrap.dedent(summary).strip().replace("\n", "")

    expected_summary = re.escape(
        textwrap.dedent(expected_summary).strip().replace("\n", "")
    )

    # In bolt v2 names are assigned by session not by model, and so which numbers
    # are assigned to each op/tensor is not deterministic based on which tests run.
    expected_summary = expected_summary.replace("NUM", R"\d+")

    assert re.match(expected_summary, summary)


def verify_model_summary(config, params, input_dims, expected_summary):
    from thirdai import deployment

    CONFIG_FILE = "./model_config"

    deployment.dump_config(json.dumps(config), CONFIG_FILE)

    model = deployment.load_model_from_config(
        config_file=CONFIG_FILE,
        parameters=params,
        input_dims=input_dims,
    )

    os.remove(CONFIG_FILE)

    compare_summaries(model, expected_summary)


@pytest.mark.unit
def test_load_model_from_config():
    config = get_config(have_user_specified_parameters=True)

    expected_summary = """
    ===================== Model =====================
    Input(input_NUM) -> tensor_NUM
    FullyConnected(fc_NUM): tensor_NUM -> tensor_NUM [dim=10, sparsity=1, activation=Tanh]
    FullyConnected(fc_NUM): tensor_NUM -> tensor_NUM [dim=20, sparsity=0.25, activation=ReLU, sampling=(random, rebuild_hash_tables=4, reconstruct_hash_functions=100)]
    FullyConnected(fc_NUM): tensor_NUM -> tensor_NUM [dim=30, sparsity=0.3, activation=Tanh, sampling=(hash_function=DWTA, permutations= 185, binsize= 8, hashes_per_table= 3, num_tables=154, range=512, reservoir_size=4, rebuild_hash_tables=4, reconstruct_hash_functions=100)]
    FullyConnected(fc_NUM): tensor_NUM -> tensor_NUM [dim=50, sparsity=0.1, activation=Softmax, sampling=(hash_function=DWTA, permutations= 8, binsize= 8, hashes_per_table= 2, num_tables=4, range=64, reservoir_size=10, rebuild_hash_tables=4, reconstruct_hash_functions=100)]
    =================================================
    """

    verify_model_summary(
        config=config,
        params={"use_sparsity": "sparse", "act": "tanh", "output_dim": 50},
        input_dims=[100],
        expected_summary=expected_summary,
    )


@pytest.mark.unit
def test_embedding_layer_config():
    config = {
        "inputs": ["input"],
        "nodes": [
            {
                "name": "emb",
                "type": "embedding",
                "num_embedding_lookups": 4,
                "lookup_size": 8,
                "log_embedding_block_size": 10,
                "reduction": "concat",
                "num_tokens_per_input": 5,
                "predecessor": "input",
            },
            {
                "name": "fc",
                "type": "fully_connected",
                "dim": 10,
                "sparsity": 1.0,
                "activation": "softmax",
                "predecessor": "emb",
            },
        ],
        "output": "fc",
        "loss": "CategoricalCrossEntropyLoss",
    }

    expected_summary = """
    ===================== Model =====================
    Input(input_NUM) -> tensor_NUM
    Embedding(emb_NUM): tensor_NUM -> tensor_NUM [ num_embedding_lookups=4, lookup_size=8, log_embedding_block_size=10, reduction=concatenation, num_tokens_per_input=5]
    FullyConnected(fc_NUM): tensor_NUM -> tensor_NUM [dim=10, sparsity=1, activation=Softmax]
    =================================================
    """

    verify_model_summary(
        config=config,
        params={},
        input_dims=[100],
        expected_summary=expected_summary,
    )


@pytest.mark.unit
def test_udt_model_config_override():
    from thirdai import deployment

    CONFIG_FILE = "./model_config"

    deployment.dump_config(json.dumps(get_config()), CONFIG_FILE)

    udt_model = bolt.UniversalDeepTransformer(
        data_types={"col": bolt.types.categorical()},
        target="col",
        n_target_classes=40,
        model_config=CONFIG_FILE,
    )
    os.remove(CONFIG_FILE)

    expected_summary = """
    ===================== Model =====================
    Input(input_NUM) -> tensor_NUM
    FullyConnected(fc_NUM): tensor_NUM -> tensor_NUM [dim=10, sparsity=1, activation=Tanh]
    FullyConnected(fc_NUM): tensor_NUM -> tensor_NUM [dim=20, sparsity=0.5, activation=ReLU, sampling=(random, rebuild_hash_tables=4, reconstruct_hash_functions=100)]
    FullyConnected(fc_NUM): tensor_NUM -> tensor_NUM [dim=30, sparsity=0.3, activation=ReLU, sampling=(hash_function=DWTA, permutations= 185, binsize= 8, hashes_per_table= 3, num_tables=154, range=512, reservoir_size=4, rebuild_hash_tables=4, reconstruct_hash_functions=100)]
    FullyConnected(fc_NUM): tensor_NUM -> tensor_NUM [dim=40, sparsity=0.1, activation=Softmax, sampling=(hash_function=DWTA, permutations= 8, binsize= 8, hashes_per_table= 2, num_tables=4, range=64, reservoir_size=10, rebuild_hash_tables=4, reconstruct_hash_functions=100)]
    =================================================
    """

    compare_summaries(udt_model._get_model(), expected_summary)


@pytest.mark.unit
def test_config_dump_load():
    from thirdai import deployment

    config = get_config()

    CONFIG_FILE = "./simple_model_config"

    deployment.dump_config(json.dumps(config), CONFIG_FILE)

    assert json.loads(deployment.load_config(CONFIG_FILE)) == config

    os.remove(CONFIG_FILE)


@pytest.mark.unit
def test_config_encryption():
    from thirdai import deployment

    config_str = json.dumps(get_config())

    CONFIG_FILE = "./encrypted_model_config"

    deployment.dump_config(config_str, CONFIG_FILE)

    with open(CONFIG_FILE, "rb") as file:
        encrypted_config_str = file.read()
        assert len(encrypted_config_str) == len(config_str)
        assert encrypted_config_str != config_str

    os.remove(CONFIG_FILE)
