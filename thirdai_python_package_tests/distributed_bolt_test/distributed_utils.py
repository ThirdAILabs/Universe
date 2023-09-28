import os
import shutil

import numpy as np
import pytest
from thirdai import bolt, dataset
from thirdai.demos import download_mnist_dataset


def get_udt_cold_start_model(n_target_classes):
    model = bolt.UniversalDeepTransformer(
        data_types={
            "QUERY": bolt.types.text(),
            "PRODUCT_ID": bolt.types.categorical(),
        },
        target="PRODUCT_ID",
        n_target_classes=n_target_classes,
        integer_target=True,
    )
    return model


def split_into_2(
    file_to_split, destination_file_1, destination_file_2, with_header=False
):
    with open(file_to_split, "r") as input_file:
        with open(destination_file_1, "w+") as f_1:
            with open(destination_file_2, "w+") as f_2:
                for i, line in enumerate(input_file):
                    if with_header and i == 0:
                        f_1.write(line)
                        f_2.write(line)
                        continue

                    if i % 2 == 0:
                        f_1.write(line)
                    else:
                        f_2.write(line)


def compare_parameters_of_two_models(model_node_1, model_node_2, atol=1e-8):
    nodes_1 = model_node_1.nodes()
    nodes_2 = model_node_2.nodes()
    for layer_1, layer_2 in zip(nodes_1, nodes_2):
        if hasattr(layer_1, "weights"):
            assert np.allclose(layer_1.weights.get(), layer_2.weights.get(), atol=atol)
        if hasattr(layer_1, "biases"):
            assert np.allclose(layer_1.biases.get(), layer_2.biases.get(), atol=atol)


def check_models_are_same_on_first_two_nodes(distributed_model):
    model_node_1 = distributed_model.get_model(worker_id=0)
    model_node_2 = distributed_model.get_model(worker_id=1)

    compare_parameters_of_two_models(model_node_1, model_node_2)


def gen_numpy_training_data(
    n_classes=10,
    n_samples=1000,
    noise_std=0.1,
    convert_to_bolt_dataset=True,
    batch_size_for_conversion=64,
):
    possible_one_hot_encodings = np.eye(n_classes)
    labels = np.random.choice(n_classes, size=n_samples).astype("uint32")
    examples = possible_one_hot_encodings[labels]
    noise = np.random.normal(0, noise_std, examples.shape)
    examples = (examples + noise).astype("float32")
    if convert_to_bolt_dataset:
        examples = dataset.from_numpy(examples, batch_size=batch_size_for_conversion)
        labels = dataset.from_numpy(labels, batch_size=batch_size_for_conversion)
    return examples, labels


def check_model_parameters_equal(model_0, model_1):
    for op_0, op_1 in zip(model_0.ops(), model_1.ops()):
        assert np.allclose(op_0.weights, op_1.weights)
        assert np.allclose(op_0.biases, op_1.biases)


def get_bolt_model():
    from thirdai import bolt

    n_classes = 10
    input_layer = bolt.nn.Input(dim=n_classes)

    hidden_layer = bolt.nn.FullyConnected(
        dim=20000,
        input_dim=n_classes,
        sparsity=0.01,
        activation="relu",
        rebuild_hash_tables=12,
        reconstruct_hash_functions=40,
    )(input_layer)
    output = bolt.nn.FullyConnected(
        dim=n_classes, input_dim=20000, activation="softmax"
    )(hidden_layer)

    labels = bolt.nn.Input(dim=n_classes)
    loss = bolt.nn.losses.CategoricalCrossEntropy(output, labels)

    model = bolt.nn.Model(inputs=[input_layer], outputs=[output], losses=[loss])

    return model


def copy_file_or_folder(source_path, destination_path):
    try:
        if os.path.isfile(source_path):
            shutil.copy2(source_path, destination_path)
            print(f"File '{source_path}' copied to '{destination_path}' successfully.")
        elif os.path.isdir(source_path):
            shutil.copytree(source_path, destination_path)
            print(
                f"Folder '{source_path}' copied to '{destination_path}' successfully."
            )
        else:
            print(f"Source '{source_path}' does not exist.")
    except PermissionError:
        print(f"Permission denied while copying '{source_path}'.")


def setup_ray(num_workers=2):
    import ray
    import thirdai.distributed_bolt as dist
    from ray.train import ScalingConfig

    # reserve one CPU for Ray Trainer
    num_cpu_per_node = (dist.get_num_cpus() - 1) // 2

    assert num_cpu_per_node >= 1, "Number of CPUs per node should be greater than 0"
    working_dir = os.path.dirname(os.path.realpath(__file__))

    ray.init(
        runtime_env={
            "working_dir": working_dir,
            "env_vars": {"OMP_NUM_THREADS": f"{num_cpu_per_node}"},
        },
        ignore_reinit_error=True,
    )
    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=False,
        resources_per_worker={"CPU": num_cpu_per_node},
        placement_strategy="PACK",
    )
    return scaling_config


def extract_metrics_from_file(filename):
    import json

    # Read the metrics dictionary from the JSON file
    with open(filename, "r") as file:
        data = json.load(file)
    return data


def write_metrics_to_file(filename, metrics):
    import json

    # Write the metrics dictionary to the file in JSON format
    with open(filename, "w") as file:
        json.dump(metrics, file)
