import os

import pytest
from distributed_utils import ray_two_node_cluster_config, remove_files
from thirdai import bolt

import pandas as pd

catalog_file = "amazon-kaggle-product-catalog.csv"


def setup_module():
    os.system(
        "curl -L https://www.dropbox.com/s/tf7e5m0cikhcb95/amazon-kaggle-product-catalog-sampled-0.05.csv?dl=0 -o amazon-kaggle-product-catalog.csv"
    )


def get_udt_cold_start_model():
    df = pd.read_csv(f"{os.getcwd()}/{catalog_file}")

    model = bolt.UniversalDeepTransformer(
        data_types={
            "QUERY": bolt.types.text(),
            "PRODUCT_ID": bolt.types.categorical(),
        },
        target="PRODUCT_ID",
        n_target_classes=df.shape[0],
        integer_target=True,
    )
    return model


# `ray_two_node_cluster_config` fixture added as parameter to start the mini_cluster
def test_distributed_cold_start(ray_two_node_cluster_config):
    udt_model = get_udt_cold_start_model()

    class FinalMetricCallback(bolt.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.ending_train_metric = 0
            self.count = 0

        def on_batch_begin(self, model, train_state):
            self.count += 1

        def on_train_end(self, model, train_state):
            self.ending_train_metric = train_state.get_train_metric_values(
                "categorical_accuracy"
            )[-1]

    final_metric = FinalMetricCallback()

    udt_model.cold_start_distributed(
        cluster_config=ray_two_node_cluster_config("linear"),
        filenames=[f"{os.getcwd()}/xaa", f"{os.getcwd()}/xab"],
        strong_column_names=["TITLE"],
        weak_column_names=["DESCRIPTION", "BULLET_POINTS", "BRAND"],
        learning_rate=0.001,
        epochs=1,
        metrics=["categorical_accuracy"],
        callbacks=[final_metric],
    )

    print(final_metric.ending_train_metric)
    assert final_metric.ending_train_metric > 0.7


def test_serializability():
    from ray.util import inspect_serializability

    class FinalMetricCallback(bolt.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.ending_train_metric = 0

        def on_train_end(self, model, train_state):
            self.ending_train_metric = train_state.get_train_metric_values(
                "categorical_accuracy"
            )[-1]

    print(inspect_serializability(FinalMetricCallback))
