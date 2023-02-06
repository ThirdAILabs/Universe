import json
import os
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
                    "reservoir_size": 10,
                },
                "predecessor": "fc_3",
            },
        ],
        "output": "fc_4",
        "loss": "CategoricalCrossEntropyLoss",
    }

    return config


@pytest.mark.unit
def test_load_model_from_config():
    from thirdai import deployment

    CONFIG_FILE = "./model_config"

    config = get_config(have_user_specified_parameters=True)

    deployment.dump_config(json.dumps(config), CONFIG_FILE)

    model = deployment.load_model_from_config(
        config_file=CONFIG_FILE,
        parameters={"use_sparsity": "sparse", "act": "tanh", "output_dim": 50},
        input_dims=[100],
    )

    summary = model.summary(detailed=True, print=False)

    expected_summary = """
    ======================= Bolt Model =======================
    input_1 (Input): dim=100
    input_1 -> fc_1 (FullyConnected): dim=10, sparsity=1, act_func=Tanh
    fc_1 -> fc_2 (FullyConnected): dim=20, sparsity=0.25, act_func=ReLU (using random sampling)
    fc_2 -> fc_3 (FullyConnected): dim=30, sparsity=0.3, act_func=Tanh (hash_function=DWTA, num_tables=154, range=512, reservoir_size=4)
    fc_3 -> fc_4 (FullyConnected): dim=50, sparsity=0.1, act_func=Softmax (hash_function=DWTA, num_tables=4, range=64, reservoir_size=10)
    ============================================================
    """

    assert textwrap.dedent(summary).strip() == textwrap.dedent(expected_summary).strip()

    os.remove(CONFIG_FILE)


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

    summary = udt_model._get_model().summary(detailed=True, print=False)

    expected_summary = """
    ======================= Bolt Model =======================
    input_1 (Input): dim=100000
    input_1 -> fc_1 (FullyConnected): dim=10, sparsity=1, act_func=Tanh
    fc_1 -> fc_2 (FullyConnected): dim=20, sparsity=0.5, act_func=ReLU (using random sampling)
    fc_2 -> fc_3 (FullyConnected): dim=30, sparsity=0.3, act_func=ReLU (hash_function=DWTA, num_tables=154, range=512, reservoir_size=4)
    fc_3 -> fc_4 (FullyConnected): dim=40, sparsity=0.1, act_func=Softmax (hash_function=DWTA, num_tables=4, range=64, reservoir_size=10)
    ============================================================
    """

    assert textwrap.dedent(summary).strip() == textwrap.dedent(expected_summary).strip()

    os.remove(CONFIG_FILE)


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
