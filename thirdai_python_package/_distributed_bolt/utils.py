from thirdai._thirdai import bolt, dataset
from typing import Tuple, Any, Optional, Dict, List
import logging


def load_dataset(
    config: Dict[str, Any], total_nodes, training_partition_data_id
):
    """
    Returns datasets as boltdatasets


    Arguments:
        config: Configuration file for the training
        total_nodes: Total number of nodes to train on.
        id: Id of the node, which want the dataset
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


def make_layers_from_config(configs: List[Dict[str, Any]]) -> List[bolt.FullyConnected]:
    """
    Returns Bolt's Fully Connected Network

    Arguments:
        configs: Configuration file for training
    """
    layers = []
    for config in configs:

        if config.get("use_default_sampling", False):
            layer = bolt.FullyConnected(
                dim=config.get("dim"),
                sparsity=config.get("sparsity", 1.0),
                activation_function=bolt.getActivationFunction(
                    config.get("activation")
                ),
            )
        else:
            layer = bolt.FullyConnected(
                dim=config.get("dim"),
                sparsity=config.get("sparsity", 1.0),
                activation_function=bolt.getActivationFunction(
                    config.get("activation")
                ),
                sampling_config=bolt.SamplingConfig(
                    hashes_per_table=config.get("hashes_per_table", 0),
                    num_tables=config.get("num_tables", 0),
                    range_pow=config.get("range_pow", 0),
                    reservoir_size=config.get("reservoir_size", 128),
                    hash_function=config.get("hash_function", "DWTA"),
                ),
            )

        layers.append(layer)
    return layers


def init_logging(logger_file: str):
    """
    Returns logger from a logger file
    """
    # Logger Init
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
    """Returns the number of CPUs present on the machine

    Returns:
        _type_: None
    """
    try:
        import multiprocessing

        return multiprocessing.cpu_count()
    except (ImportError):
        print("Could not find num_cpus, setting num_cpus to DEFAULT=1")
        return 1
