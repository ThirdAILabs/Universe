import numpy as np
import pandas as pd
import pytest
from download_dataset_fixtures import download_yelp_chi_dataset
from sklearn import metrics
from sklearn.model_selection import train_test_split
from thirdai import bolt

pytestmark = [pytest.mark.unit]


@pytest.fixture(scope="module")
def train_udt_on_yelp_chi(download_yelp_chi_dataset):
    (
        train_data_path,
        eval_data_path,
        inference_samples,
        data_types,
    ) = download_yelp_chi_dataset

    ground_truth = [inference_sample[1] for inference_sample in inference_samples]

    model = bolt.UniversalDeepTransformer(
        data_types=data_types,
        target="target",
        # Turn off pairgrams to make the test fast (~1 min on my m1)
        contextual_columns=False,
    )

    # We need to index these nodes because the model needs to know about them
    # for training: nodes in the train set have neighbors in the eval set, and
    # the model uses neighbor features. The eval set has all labels set to 0,
    # so this will not leak testing information.
    model.index_nodes(eval_data_path)

    auc = None
    for _ in range(15):
        train_metrics = model.train(
            train_data_path,
            learning_rate=0.001,
            epochs=1,
            metrics=["categorical_accuracy"],
        )
        assert train_metrics["train_categorical_accuracy"][-1] > 0

        activations = model.predict_batch([x[0] for x in inference_samples])
        auc = metrics.roc_auc_score(ground_truth, activations[:, 1])
        print("AUC: ", auc)

    return model, auc


def test_udt_yelp_chi_accuracy(train_udt_on_yelp_chi):
    # Gets around 0.91
    _, auc = train_udt_on_yelp_chi
    assert auc > 0.89


def test_udt_yelp_chi_save_load(train_udt_on_yelp_chi, download_yelp_chi_dataset):
    _, _, inference_samples, _ = download_yelp_chi_dataset
    model, auc = train_udt_on_yelp_chi

    ground_truth = [inference_sample[1] for inference_sample in inference_samples]
    model.save("udt_graph.serialized")
    model = bolt.UniversalDeepTransformer.load("udt_graph.serialized")
    activations = model.predict_batch([x[0] for x in inference_samples])
    new_auc = metrics.roc_auc_score(ground_truth, activations[:, 1])
    assert new_auc == auc


def test_udt_yelp_chi_predictions(train_udt_on_yelp_chi, download_yelp_chi_dataset):
    _, _, inference_samples, _ = download_yelp_chi_dataset
    model, auc = train_udt_on_yelp_chi

    single_predictions = []
    for sample, _ in inference_samples:
        prediction = model.predict(sample)
        single_predictions.append(prediction)
    single_predictions = np.array(single_predictions)

    batch_size = 64
    batch_predictions = []
    for batch_start in range(0, len(inference_samples), batch_size):
        predictions = model.predict_batch(
            [
                sample[0]
                for sample in inference_samples[batch_start : batch_start + batch_size]
            ]
        )
        batch_predictions += list(predictions)
    batch_predictions = np.array(batch_predictions)

    assert np.array_equal(batch_predictions, single_predictions)

    ground_truth = [inference_sample[1] for inference_sample in inference_samples]
    single_prediction_auc = metrics.roc_auc_score(
        ground_truth, single_predictions[:, 1]
    )
    assert auc == single_prediction_auc


def get_no_features_gnn(num_classes):
    return bolt.UniversalDeepTransformer(
        data_types={
            "node_id": bolt.types.node_id(),
            "target": bolt.types.categorical(n_classes=num_classes, type="int"),
            "neighbors": bolt.types.neighbors(),
            "feature": bolt.types.numerical(range=(0, 2)),
        },
        target="target",
    )


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
        df["feature"] = np.ones(shape=chunk_size)
        df.to_csv(f"graph_chunk_{chunk}.csv", index=False)
        model.train(f"graph_chunk_{chunk}.csv", epochs=1)

    chunk_id_to_test = 7
    old_accuracy = model.evaluate(
        f"graph_chunk_{chunk_id_to_test}.csv", metrics=["categorical_accuracy"]
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
        f"graph_chunk_{chunk_id_to_test}.csv", metrics=["categorical_accuracy"]
    )
    assert (
        old_accuracy["val_categorical_accuracy"][-1]
        == new_accuracy["val_categorical_accuracy"][-1]
    )


def test_no_neighbors_causes_no_errors():
    num_classes = 2
    model = get_no_features_gnn(num_classes)

    df = pd.DataFrame()
    df["node_id"] = np.arange(0, 100)
    df["target"] = np.full(shape=100, fill_value=1)
    df["neighbors"] = ["" for _ in range(100)]
    df["feature"] = np.ones(shape=100)

    df.to_csv(f"boring_graph.csv", index=False)
    model.train("boring_graph.csv")

    model.evaluate("boring_graph.csv", metrics=["categorical_accuracy"])
