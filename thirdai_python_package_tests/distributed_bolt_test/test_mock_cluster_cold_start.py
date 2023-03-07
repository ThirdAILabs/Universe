import os

import pandas as pd
import pytest
from distributed_utils import ray_two_node_cluster_config, split_into_2
from thirdai import bolt

catalog_file = "amazon-kaggle-product-catalog.csv"

pytestmark = [pytest.mark.distributed]


def setup_module():
    import os

    path = "amazon_product_catalog"
    if not os.path.exists(path):
        os.makedirs(path)

    if not os.path.exists("amazon-kaggle-product-catalog.csv"):
        os.system(
            "curl -L https://www.dropbox.com/s/tf7e5m0cikhcb95/amazon-kaggle-product-catalog-sampled-0.05.csv?dl=0 -o amazon-kaggle-product-catalog.csv"
        )
    if not os.path.exists(f"{path}/part1") or not os.path.exists(f"{path}/part2"):
        split_into_2(
            file_to_split="amazon-kaggle-product-catalog.csv",
            destination_file_1=f"{path}/part1",
            destination_file_2=f"{path}/part2",
            with_header=True,
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

    metrics = udt_model.cold_start_distributed(
        cluster_config=ray_two_node_cluster_config("linear"),
        filenames=[
            f"{os.getcwd()}/amazon_product_catalog/part1",
            f"{os.getcwd()}/amazon_product_catalog/part2",
        ],
        strong_column_names=["TITLE"],
        weak_column_names=["DESCRIPTION", "BULLET_POINTS", "BRAND"],
        batch_size=2048,
        learning_rate=0.001,
        epochs=5,
        metrics=["categorical_accuracy"],
    )
    train_metrics = metrics["train_metrics"]
    overall_metrics = {}
    for metrics_per_node in train_metrics:
        for key, value in metrics_per_node.items():
            print(value)
            if key not in overall_metrics:
                overall_metrics[key] = 0
            # Here we are averaging the metrics, hence divding the
            # metric "categorical_accuracy" by 2.
            overall_metrics[key] += value[-1] / 2

    assert overall_metrics["categorical_accuracy"] > 0.7
