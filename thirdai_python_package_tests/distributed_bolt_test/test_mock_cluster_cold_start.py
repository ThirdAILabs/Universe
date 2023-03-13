import os

import pandas as pd
import pytest
from distributed_utils import ray_two_node_cluster_config, split_into_2
from thirdai import bolt
from thirdai.demos import (
    download_amazon_kaggle_product_catalog_sampled as download_amazon_kaggle_product_catalog_sampled_wrapped,
)


@pytest.fixture(scope="module")
def download_amazon_kaggle_product_catalog_sampled():
    return download_amazon_kaggle_product_catalog_sampled_wrapped()


pytestmark = [pytest.mark.distributed]


def download_and_split_dataset(download_amazon_kaggle_product_catalog_sampled):
    import os

    path = "amazon_product_catalog"
    if not os.path.exists(path):
        os.makedirs(path)

    catalog_file, n_target_classes = download_amazon_kaggle_product_catalog_sampled

    if not os.path.exists(f"{path}/part1") or not os.path.exists(f"{path}/part2"):
        split_into_2(
            file_to_split=catalog_file,
            destination_file_1=f"{path}/part1",
            destination_file_2=f"{path}/part2",
            with_header=True,
        )
    return n_target_classes


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


# `ray_two_node_cluster_config` fixture added as parameter to start the mini_cluster
def test_distributed_cold_start(
    ray_two_node_cluster_config, download_amazon_kaggle_product_catalog_sampled
):
    n_target_classes = download_and_split_dataset(
        download_amazon_kaggle_product_catalog_sampled
    )

    udt_model = get_udt_cold_start_model(n_target_classes)

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
            if key not in overall_metrics:
                overall_metrics[key] = 0
            # Here we are averaging the metrics, hence divding the
            # metric "categorical_accuracy" by 2.
            overall_metrics[key] += value[-1] / 2

    assert overall_metrics["categorical_accuracy"] > 0.7
