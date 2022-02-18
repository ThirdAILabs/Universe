import toml
import sys
from thirdai import bolt, dataset
import numpy as np
from sklearn.metrics import roc_auc_score
from typing import MutableMapping, List, Tuple, Any, Union, Optional


def create_fully_connected_layer_configs(
    configs: List[MutableMapping[str, Any]]
) -> List[bolt.LayerConfig]:
    layers = []
    for config in configs:
        layer = bolt.LayerConfig(
            dim=config.get("dim"),
            load_factor=config.get("sparsity", 1.0),
            activation_function=config.get("activation"),
            sampling_config=bolt.SamplingConfig(
                hashes_per_table=config.get("hashes_per_table", 0),
                num_tables=config.get("num_tables", 0),
                range_pow=config.get("range_pow", 0),
                reservoir_size=config.get("reservoir_size", 128),
            ),
        )
        layers.append(layer)
    return layers


def create_embedding_layer_config(
    config: MutableMapping[str, Any]
) -> bolt.EmbeddingLayerConfig:
    return bolt.EmbeddingLayerConfig(
        num_embedding_lookups=config.get("num_embedding_lookups"),
        lookup_size=config.get("lookup_size"),
        log_embedding_block_size=config.get("log_embedding_block_size"),
    )


AnyBoltDataset = Union[dataset.InMemorySparseDataset,
                       dataset.InMemoryDenseDataset]


def load_dataset(
    config: MutableMapping[str, Any]
) -> Optional[Tuple[AnyBoltDataset, AnyBoltDataset]]:
    train_filename = config["dataset"]["train_data"]
    test_filename = config["dataset"]["test_data"]
    batch_size = config["params"]["batch_size"]
    if config["dataset"]["format"].lower() == "svm":
        train = dataset.load_svm_dataset(train_filename, batch_size)
        test = dataset.load_svm_dataset(test_filename, batch_size)
        return train, test
    elif config["dataset"]["format"].lower() == "csv":
        delimiter = config["dataset"].get("delimeter", ",")
        train = dataset.load_csv_dataset(train_filename, batch_size, delimiter)
        test = dataset.load_csv_dataset(test_filename, batch_size, delimiter)
        return train, test
    else:
        print("Invalid dataset format specified")
        return None


def load_click_through_dataset(
    config: MutableMapping[str, Any]
) -> Tuple[dataset.ClickThroughDataset, dataset.ClickThroughDataset]:
    train_filename = config["dataset"]["train_data"]
    test_filename = config["dataset"]["test_data"]
    batch_size = config["params"]["batch_size"]
    dense_features = config["dataset"]["dense_features"]
    categorical_features = config["dataset"]["categorical_features"]
    train = dataset.load_click_through_dataset(
        train_filename, batch_size, dense_features, categorical_features
    )
    test = dataset.load_click_through_dataset(
        test_filename, batch_size, dense_features, categorical_features
    )
    return train, test


def get_labels(dataset: str) -> np.array:
    labels = []
    with open(dataset) as file:
        for line in file.readlines():
            items = line.strip().split()
            label = int(items[0])
            labels.append(label)
    return np.array(labels)


def train_fcn(config: MutableMapping[str, Any]):
    layers = create_fully_connected_layer_configs(config["layers"])
    input_dim = config["dataset"]["input_dim"]
    network = bolt.Network(layers=layers, input_dim=input_dim)

    learning_rate = config["params"]["learning_rate"]
    epochs = config["params"]["epochs"]
    max_test_batches = config["dataset"].get("max_test_batches", None)
    rehash = config["params"]["rehash"]
    rebuild = config["params"]["rebuild"]

    data = load_dataset(config)
    if data is None:
        return
    train, test = data

    for _ in range(epochs):
        network.train(train, learning_rate, 1, rehash, rebuild)
        if max_test_batches is None:
            network.predict(test)
        else:
            network.predict(test, max_test_batches)
    if not max_test_batches is None:
        network.predict(test)


def train_dlrm(config: MutableMapping[str, Any]):
    embedding_layer = create_embedding_layer_config(config["embedding_layer"])
    bottom_mlp = create_fully_connected_layer_configs(
        config["bottom_mlp_layers"])
    top_mlp = create_fully_connected_layer_configs(config["top_mlp_layers"])
    input_dim = config["dataset"]["dense_features"]
    dlrm = bolt.DLRM(
        embedding_layer=embedding_layer,
        bottom_mlp=bottom_mlp,
        top_mlp=top_mlp,
        input_dim=input_dim,
    )

    learning_rate = config["params"]["learning_rate"]
    epochs = config["params"]["epochs"]
    rehash = config["params"]["rehash"]
    rebuild = config["params"]["rebuild"]

    use_auc = config["params"].get("use_auc", False)

    train, test = load_click_through_dataset(config)
    labels = get_labels(config["dataset"]["test_data"])

    for _ in range(epochs):
        dlrm.train(train, learning_rate, 1, rehash, rebuild)
        scores = dlrm.predict(test)
        preds = np.argmax(scores, axis=1)
        acc = np.mean(preds == labels)
        print("Accuracy: ", acc)
        if use_auc:
            auc = roc_auc_score(labels, scores)
            print("AUC: ", auc)


def is_dlrm(config: MutableMapping[str, Any]) -> bool:
    return "bottom_mlp_layers" in config.keys() and "top_mlp_layers" in config.keys()


def is_fcn(config: MutableMapping[str, Any]) -> bool:
    return "layers" in config.keys()


def main():
    config = toml.load(sys.argv[1])
    if is_fcn(config):
        train_fcn(config)
    elif is_dlrm(config):
        train_dlrm(config)
    else:
        print("Invalid network architecture specified")


if __name__ == "__main__":
    main()
