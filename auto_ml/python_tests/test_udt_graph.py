import numpy as np
import pandas as pd
import pytest
from download_dataset_fixtures import download_yelp_chi_dataset
from sklearn import metrics
from sklearn.model_selection import train_test_split
from thirdai import bolt

pytestmark = [pytest.mark.unit]


def get_no_features_gnn(num_classes):
    return bolt.UniversalDeepTransformer(
        data_types={
            "node_id": bolt.types.node_id(),
            "target": bolt.types.categorical(),
            "neighbors": bolt.types.neighbors(),
        },
        target="target",
        n_target_classes=num_classes,
        integer_target=True,
    )


def test_udt_on_yelp_chi(download_yelp_chi_dataset):
    all_data = pd.read_csv("yelp_all.csv")
    numerical_col_names = ["col_" + str(i) for i in range(32)]
    numerical_col_ranges = (
        all_data[numerical_col_names].agg([min, max]).T.values.tolist()
    )

    train_data, test_data = train_test_split(all_data, test_size=0.5)

    train_data = train_data.sample(frac=1)
    train_data.to_csv("yelp_train.csv", index=False)

    ground_truth = test_data["target"].to_numpy()
    test_data["target"] = np.zeros(len(ground_truth))
    test_data.to_csv("yelp_test.csv", index=False)

    model = bolt.UniversalDeepTransformer(
        data_types={
            "node_id": bolt.types.node_id(),
            **{
                col_name: bolt.types.numerical(col_range)
                for col_range, col_name in zip(
                    numerical_col_ranges, numerical_col_names
                )
            },
            "target": bolt.types.categorical(),
            "neighbors": bolt.types.neighbors(),
        },
        target="target",
        n_target_classes=2,
        integer_target=True,
        # Turn off pairgrams to make the test fast (~1 min on my m1)
        options={"contextual_columns": False},
    )

    # We need to index these nodes because the model needs to know about them
    # for training: nodes in the train set have neighbors in the test set, and
    # the model uses neighbor features.
    model.index_nodes("yelp_test.csv")

    for epoch in range(15):
        train_metrics = model.train(
            "yelp_train.csv",
            learning_rate=0.001,
            epochs=1,
            metrics=["categorical_accuracy"],
        )
        assert train_metrics["categorical_accuracy"][-1] > 0

        activations = model.evaluate("yelp_test.csv")
        auc = metrics.roc_auc_score(ground_truth, activations[:, 1])
        print("AUC: ", auc)

    # Gets around 0.91
    assert auc > 0.89

    model.save("udt_graph.serialized")
    model = bolt.UniversalDeepTransformer.load("udt_graph.serialized")
    activations = model.evaluate("yelp_test.csv")
    new_auc = metrics.roc_auc_score(ground_truth, activations[:, 1])
    assert new_auc == auc


def test_graph_clearing_and_indexing():
    num_chunks = 10
    chunk_size = 100
    num_classes = 2

    model = get_no_features_gnn(num_classes)

    # The graph is linear, so each node is connected to its predecessor. Node
    # id i has class id = (i / chunk_size) % num_classes.
    for chunk in range(num_chunks):
        chunk_start = chunk_size * chunk
        chunk_end = chunk_start + chunk_size
        df = pd.DataFrame()
        df["node_id"] = np.arange(chunk_start, chunk_end)
        df["target"] = np.full(shape=chunk_size, fill_value=chunk % num_classes)
        df["neighbors"] = [
            max(0, node_id - 1) for node_id in range(chunk_start, chunk_end)
        ]
        df.to_csv(f"graph_chunk_{chunk}.csv", index=False)
        model.train(f"graph_chunk_{chunk}.csv", epochs=1)

    chunk_id_to_test = 7
    old_accuracy = model.evaluate(
        f"graph_chunk_{chunk_id_to_test}.csv",
        metrics=["categorical_accuracy"],
        return_metrics=True,
    )
    model.clear_graph()
    with pytest.raises(
        RuntimeError,
        match=r"The model's stored graph is in an unexpected state: No feature vector currently stored for node.*",
    ):
        model.evaluate(
            f"graph_chunk_{chunk_id_to_test}.csv", metrics=["categorical_accuracy"]
        )
    for chunk_id in range(chunk_id_to_test):
        model.index_nodes(f"graph_chunk_{chunk_id}.csv")
    new_accuracy = model.evaluate(
        f"graph_chunk_{chunk_id_to_test}.csv",
        metrics=["categorical_accuracy"],
        return_metrics=True,
    )
    assert old_accuracy["categorical_accuracy"] == new_accuracy["categorical_accuracy"]


def test_no_neighbors_causes_no_errors():
    num_classes = 2
    model = get_no_features_gnn(num_classes)

    df = pd.DataFrame()
    df["node_id"] = np.arange(0, 100)
    df["target"] = np.full(shape=100, fill_value=1)
    df["neighbors"] = ["" for _ in range(100)]

    df.to_csv(f"boring_graph.csv", index=False)
    model.train("boring_graph.csv")

    model.evaluate(
        "boring_graph.csv", metrics=["categorical_accuracy"], return_metrics=True
    )["categorical_accuracy"]
