from thirdai._thirdai import bolt, dataset
from typing import Tuple, Any, Optional, Dict, List
import logging


def load_train_test_data(
    config: Dict[str, Any], total_nodes, training_partition_data_id
):
    """
    Returns datasets as boltdatasets

    :param config: config dict for training
    :type config: Dict[str, Any]
    :param total_nodes: Total number of nodes to train on
    :type total_nodes: int
    :param training_partition_data_id: Id of the node, which want the dataset
    :type training_partition_data_id: int
    :raises ValueError: Invalied Dataset Format
    :return: returns training and testing data
    :rtype: [BoltDataset, BoltDataset, BoltDataset, BoltDataset]
    """
    train_filename = config["dataset"]["train_data"][training_partition_data_id]
    test_filename = config["dataset"]["test_data"]
    batch_size = int(config["params"]["batch_size"] / total_nodes)
    if config["dataset"]["format"].lower() == "svm":
        train_x, train_y = dataset.load_bolt_svm_dataset(train_filename, batch_size)
        test_x, test_y = dataset.load_bolt_svm_dataset(test_filename, batch_size)
        return train_x, train_y, test_x, test_y
    elif config["dataset"]["format"].lower() == "csv":
        delimiter = config["dataset"].get("delimeter", ",")
        train_x, train_y = dataset.load_bolt_csv_dataset(
            train_filename, batch_size, delimiter
        )
        test_x, test_y = dataset.load_bolt_csv_dataset(
            test_filename, batch_size, delimiter
        )
        return train_x, train_y, test_x, test_y
    else:
        raise ValueError("Invalid dataset format specified")


def config_get(config, field):
    if field not in config:
        raise ValueError(
            f'The field "{field}" was expected to be in "{config}" but was not found.'
        )
    return config[field]


def construct_fully_connected_node(fc_config):
    sparsity = fc_config.get("sparsity", 1)
    return bolt.graph.FullyConnected(
        dim=config_get(fc_config, "dim"),
        sparsity=sparsity,
        activation=config_get(fc_config, "activation"),
    )


def construct_node(node_config):
    node_type = config_get(node_config, "type")
    if node_type == "Input":
        return bolt.graph.Input(dim=config_get(node_config, "dim"))
    if node_type == "FullyConnected":
        return construct_fully_connected_node(node_config)
    raise ValueError(f"{node_type} is not a valid node type.")


def contruct_dag_model(config):
    name_to_node = {}

    def get_node_by_name(node_name):
        if node_name in name_to_node:
            return name_to_node[node_name]
        raise ValueError(f"{node_name} not found in previously defined nodes")

    nodes_with_no_successor = set()
    inputs = []
    for node_config in config_get(config, "nodes"):
        node = construct_node(node_config)
        node_name = config_get(node_config, "name")
        node_type = config_get(node_config, "type")
        if node_type == "Input":
            inputs.append(node)
        elif "pred" in node_config:
            pred_name = node_config["pred"]
            pred_node = get_node_by_name(pred_name)
            nodes_with_no_successor.remove(pred_name)
            node(pred_node)
        elif "preds" in node_config:
            pred_names = node_config["preds"]
            pred_nodes = [get_node_by_name(pred_name) for pred_name in pred_names]
            for pred_name in pred_names:
                nodes_with_no_successor.remove(pred_name)
            if config_get(node_config, "type") == "Switch":
                node(pred_nodes[0], pred_nodes[1])
            else:
                node(pred_nodes)
        else:
            raise ValueError(
                "Node should either be an Input/TokenInput or specify pred/preds"
            )

        nodes_with_no_successor.add(node_name)
        name_to_node[node_name] = node

    if len(nodes_with_no_successor) != 1:
        raise ValueError(
            "There should only be one output node (nodes with no successors), "
            + f"but found {len(nodes_with_no_successor)}"
        )

    output_node = name_to_node[list(nodes_with_no_successor)[0]]
    return inputs, output_node


def init_logging(logger_file: str):
    """
    Returns logger from a logger file
    """
    logger = logging.getLogger(logger_file)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(logger_file)
    formatter = logging.Formatter(
        "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def get_num_cpus():
    try:
        import multiprocessing

        return multiprocessing.cpu_count()
    except (ImportError):
        print("Could not find num_cpus, setting num_cpus to DEFAULT=1")
        return 1
