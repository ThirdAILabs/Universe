from thirdai._thirdai import bolt, dataset
from typing import Tuple, Any, Optional, Dict, List
import toml
import os
import logging
 

def find_full_filepath(filename: str) -> str:
    data_path_file = ("./dataset_paths.toml")
    prefix_table = toml.load(data_path_file)
    for prefix in prefix_table["prefixes"]:
        if os.path.exists(prefix + filename):
            return prefix + filename
    print(
        "Could not find file '"
        + filename
        + "' on any filepaths. Add correct path to 'Universe/dataset_paths.toml'"
    )
    sys.exit(1)

def load_dataset(
    config: Dict[str, Any]
    , total_nodes) -> Optional[
        Tuple[
            dataset.BoltDataset,  # train_x
            dataset.BoltDataset,  # train_y
            dataset.BoltDataset,  # test_x
            dataset.BoltDataset,  # test_y
        ]
    ]:
    train_filename = find_full_filepath(config["dataset"]["train_data"])
    test_filename = find_full_filepath(config["dataset"]["test_data"])
    batch_size = int(config["params"]["batch_size"]/total_nodes)
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
        print("Invalid dataset format specified")
        return None

def create_fully_connected_layer_configs(
    configs: List[Dict[str, Any]]
) -> List[bolt.FullyConnected]:
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


def initLogging():
    # Logger Init
    logger = logging.getLogger('DistributedBolt')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('logfile.log')
    formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger